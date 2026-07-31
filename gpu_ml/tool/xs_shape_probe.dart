/// Per-shape T sweep for the shared-x decode kernels, standalone (no model
/// load).  The --ab-rp e2e A/B showed the global row-pack retune is a net
/// loss; this isolates the right T PER SHAPE:
///   dense  2048x2048 (attn/delta projections, router)
///   gate   512x2048 x 8 slots via idxb (expfused gate/up)
///   down   2048x512 x 8 slots via idxb (expfused down)
/// Chain of 400 serialized dispatches, q8_0 bodies byte-for-byte like the
/// plan (shared-x staged, padded).
///
///   dart run tool/xs_shape_probe.dart
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:gpu_ml/gpu_ml.dart';
import 'package:minigpu/minigpu.dart';

const reps = 3, chainN = 400;

String kernel(int rows, int cols, int t, {bool slots = false}) {
  final r = 256 ~/ t;
  var body = t == 256
      ? QuantizedTensor.matVecBodyWGSL(GgmlType.q8_0)
      : QuantizedTensor.matVecBodyWGSL(GgmlType.q8_0,
          threadVar: 'trd', stride: '${t}u');
  body = QuantizedTensor.sharedXBody(body);
  final bpe = rows * cols ~/ 32 * 34;
  final reduce = t == 256
      ? '''
  workgroupBarrier();
  for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
    if (lid.x < s) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s]; }
    workgroupBarrier();
  }
'''
      : '''
  workgroupBarrier();
  for (var s: u32 = ${t ~/ 2}u; s > 0u; s = s >> 1u) {
    if (trd < s) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s]; }
    workgroupBarrier();
  }
''';
  return '''
@group(0) @binding(0) var<storage, read_write> wq: array<u32>;
@group(0) @binding(1) var<storage, read_write> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> idxb: array<u32>;

const ROWS: u32 = ${rows}u;
const COLS: u32 = ${cols}u;

${QuantizedTensor.accessorsWGSL}
${QuantizedTensor.xsDeclWGSL(cols)}

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  ${t == 256 ? '' : 'let trd: u32 = lid.x % ${t}u;'}
  let row: u32 = ${t == 256 ? 'wid.x + wid.y * nwg.x' : '(wid.x + wid.y * nwg.x) * ${r}u + lid.x / ${t}u'};
  let eb: u32 = ${slots ? 'idxb[wid.z] * ${bpe}u' : '0u'};
  var acc: f32 = 0.0;
${QuantizedTensor.stageXsWGSL(cols)}
  if (row < ROWS) {
$body
  }
  scratch[lid.x] = acc;
$reduce
  if (${t == 256 ? 'lid.x == 0u' : 'trd == 0u'} && row < ROWS) {
    y[${slots ? 'wid.z * ROWS + row' : 'row'}] = scratch[lid.x];
  }
}
''';
}

Future<void> main() async {
  final gpu = Minigpu.forAdapter('4090');
  await gpu.init();
  stdout.writeln('adapter: ${gpu.adapterName}');

  // (label, rows, cols, slots)
  final shapes = [
    ('dense 2048x2048', 2048, 2048, false),
    ('gate  512x2048x8', 512, 2048, true),
    ('down  2048x512x8', 2048, 512, true),
  ];
  const topK = 8;

  final idxb = gpu.createBuffer(topK * 4, BufferDataType.uint32);
  await idxb.write(Uint32List.fromList([0, 1, 2, 3, 4, 5, 6, 7]), topK,
      dataType: BufferDataType.uint32);
  final tmp = Float32List(4);

  for (final (label, rows, cols, slots) in shapes) {
    final bpe = rows * cols ~/ 32 * 34;
    final wBytes = slots ? bpe * topK : bpe;
    final wq = gpu.createBuffer(wBytes + 4, BufferDataType.uint32);
    final x = gpu.createBuffer((slots ? topK : 1) * cols * 4,
        BufferDataType.float32);
    final y = gpu.createBuffer((slots ? topK : 1) * rows * 4,
        BufferDataType.float32);
    for (final t in [256, 64, 32, 16]) {
      if (t < 256 && t > cols ~/ 32) continue;
      final r = 256 ~/ t;
      try {
        final s = gpu.createComputeShader();
        s.loadKernelString(kernel(rows, cols, t, slots: slots));
        s.setBuffer('wq', wq);
        s.setBuffer('x', x);
        s.setBuffer('y', y);
        s.setBuffer('idxb', idxb);
        final wgs = (rows + r - 1) ~/ r;
        s.dispatchFire(wgs, 1, slots ? topK : 1);
        await y.read(tmp, 4);
        double bestUs = 1e18;
        for (int rep = 0; rep < reps; rep++) {
          final sw = Stopwatch()..start();
          for (int i = 0; i < chainN; i++) {
            s.dispatchFire(wgs, 1, slots ? topK : 1);
          }
          await y.read(tmp, 4);
          final us = sw.elapsedMicroseconds / chainN;
          if (us < bestUs) bestUs = us;
        }
        s.destroy();
        final gbps = wBytes / (bestUs / 1e6) / 1e9;
        stdout.writeln('$label T=${t.toString().padLeft(3)} '
            'wgs=${(wgs * (slots ? topK : 1)).toString().padLeft(5)}  '
            '${bestUs.toStringAsFixed(2)} us  ${gbps.toStringAsFixed(0)} GB/s');
      } catch (e) {
        stdout.writeln('$label T=$t FAILED: ${e.toString().split('\n').first}');
      }
    }
    wq.destroy();
    x.destroy();
    y.destroy();
  }
  idxb.destroy();
  exit(0);
}

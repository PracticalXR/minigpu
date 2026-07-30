/// Kernel-body tax probe: the last suspect standing for the decode wall.
/// Synthetic loads at plan-qmv geometry (2048x2048 q8_0, T=64, 512 wgs) run
/// at ~1300 GB/s; the real kernel was historically measured at ~270 GB/s.
/// This runs the REAL codegen'd body against hand-optimized variants in a
/// 400-dispatch serialized chain (decode-realistic).
///
/// Variants:
///   A real   — byte-for-byte matVecBodyWGSL(q8_0, T=64) as the plan builds
///   F w-only — A with x loads replaced by 1.0 (isolates the x-load tax)
///   C vec4x  — A with the 32 scalar x loads -> 8 vec4 loads
///   D unpack — C with manual sign-extends -> unpack4xI8
///   E floor  — pure u32 streaming, same geometry (the hardware limit)
///
///   dart run tool/kernel_tax_probe.dart
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:gpu_ml/gpu_ml.dart';
import 'package:minigpu/minigpu.dart';

const rows = 2048, cols = 2048;
const wBytes = rows * cols ~/ 32 * 34; // 4.46 MB q8_0
const reps = 3, chainN = 400;

String wrap(String body,
        {bool vec4x = false, bool shared = false, String preBody = ''}) =>
    '''
@group(0) @binding(0) var<storage, read_write> wq: array<u32>;
@group(0) @binding(1) var<storage, read_write> ${vec4x ? 'x4: array<vec4<f32>>' : 'x: array<f32>'};
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

const ROWS: u32 = ${rows}u;
const COLS: u32 = ${cols}u;

${QuantizedTensor.accessorsWGSL}

var<workgroup> scratch: array<f32, 256>;
${shared ? 'var<workgroup> xs: array<f32, ${cols + cols ~/ 32}>;' : ''}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let trd: u32 = lid.x % 64u;
  let row: u32 = (wid.x + wid.y * nwg.x) * 4u + lid.x / 64u;
  let eb: u32 = 0u;
  var acc: f32 = 0.0;
$preBody
  if (row < ROWS) {
$body
  }
  scratch[lid.x] = acc;
  workgroupBarrier();
  for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
    if (trd < s) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s]; }
    workgroupBarrier();
  }
  if (trd == 0u && row < ROWS) { y[row] = scratch[lid.x]; }
}
''';

const optHeader = '''
  let nb: u32 = COLS / 32u;
  for (var j: u32 = trd; j < nb; j = j + 64u) {
    let base: u32 = eb + (row * nb + j) * 34u;
    let d: f32 = f16At(base);
    let qb: u32 = base + 2u;
    let qw: u32 = qb >> 2u;
    let qs: u32 = (qb & 3u) * 8u;
    var carry: u32 = wq[qw];
    var bsum: f32 = 0.0;
    let xb4: u32 = j * 8u;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
      let nxt: u32 = wq[qw + k + 1u];
      var raw: u32;
      if (qs == 0u) { raw = carry; } else { raw = (carry >> qs) | (nxt << (32u - qs)); }
      carry = nxt;
''';

const bodyVec4x = '''
$optHeader
      let xv: vec4<f32> = x4[xb4 + k];
      bsum = bsum + f32(bitcast<i32>(raw << 24u) >> 24u) * xv.x
                  + f32(bitcast<i32>(raw << 16u) >> 24u) * xv.y
                  + f32(bitcast<i32>(raw << 8u) >> 24u) * xv.z
                  + f32(bitcast<i32>(raw) >> 24u) * xv.w;
    }
    acc = acc + d * bsum;
  }
''';

const bodyUnpack = '''
$optHeader
      let xv: vec4<f32> = x4[xb4 + k];
      bsum = bsum + dot(vec4<f32>(unpack4xI8(raw)), xv);
    }
    acc = acc + d * bsum;
  }
''';

const bodyFloor = '''
  let nb: u32 = COLS / 32u;
  for (var j: u32 = trd; j < nb; j = j + 64u) {
    let w0: u32 = (row * nb + j) * 8u;
    var bsum: f32 = 0.0;
    for (var k: u32 = 0u; k < 9u; k = k + 1u) {
      bsum = bsum + f32(wq[(w0 + k) & ${(wBytes ~/ 4) - 1}u]);
    }
    acc = acc + bsum;
  }
''';

// x loads + weight loads, extracts kept, multiplies -> adds.
const bodyNoMul = '''
$optHeader
      let xv: vec4<f32> = x4[xb4 + k];
      bsum = bsum + f32(bitcast<i32>(raw << 24u) >> 24u) + xv.x
                  + f32(bitcast<i32>(raw << 16u) >> 24u) + xv.y
                  + f32(bitcast<i32>(raw << 8u) >> 24u) + xv.z
                  + f32(bitcast<i32>(raw) >> 24u) + xv.w;
    }
    acc = acc + d * bsum;
  }
''';

// x loads + weight loads + multiplies, extracts removed (raw used whole).
const bodyNoExtract = '''
$optHeader
      let xv: vec4<f32> = x4[xb4 + k];
      let rf: f32 = f32(raw);
      bsum = bsum + rf * xv.x + rf * xv.y + rf * xv.z + rf * xv.w;
    }
    acc = acc + d * bsum;
  }
''';

// Full real math, but x staged once per workgroup into PADDED shared memory
// (stride 33 kills the (k*4+c)%32 lane-independent bank conflict that
// invalidated the wave-11 shared-x ablation).
const preShared = '''
  for (var i: u32 = lid.x; i < COLS; i = i + 256u) {
    xs[i + i / 32u] = x4[i >> 2u][i & 3u];
  }
  workgroupBarrier();
''';

const bodyShared = '''
  let nb: u32 = COLS / 32u;
  for (var j: u32 = trd; j < nb; j = j + 64u) {
    let base: u32 = eb + (row * nb + j) * 34u;
    let d: f32 = f16At(base);
    let qb: u32 = base + 2u;
    let qw: u32 = qb >> 2u;
    let qs: u32 = (qb & 3u) * 8u;
    var carry: u32 = wq[qw];
    var bsum: f32 = 0.0;
    let xb: u32 = j * 32u;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
      let nxt: u32 = wq[qw + k + 1u];
      var raw: u32;
      if (qs == 0u) { raw = carry; } else { raw = (carry >> qs) | (nxt << (32u - qs)); }
      carry = nxt;
      let xi: u32 = xb + k * 4u;
      bsum = bsum + f32(bitcast<i32>(raw << 24u) >> 24u) * xs[xi + xi / 32u]
                  + f32(bitcast<i32>(raw << 16u) >> 24u) * xs[xi + 1u + (xi + 1u) / 32u]
                  + f32(bitcast<i32>(raw << 8u) >> 24u) * xs[xi + 2u + (xi + 2u) / 32u]
                  + f32(bitcast<i32>(raw) >> 24u) * xs[xi + 3u + (xi + 3u) / 32u];
    }
    acc = acc + d * bsum;
  }
''';

Future<void> main() async {
  final gpu = Minigpu.forAdapter('4090');
  await gpu.init();
  stdout.writeln('adapter: ${gpu.adapterName}');

  final wq = gpu.createBuffer(wBytes + 4, BufferDataType.uint32);
  final x = gpu.createBuffer(cols * 4, BufferDataType.float32);
  final y = gpu.createBuffer(rows * 4, BufferDataType.float32);
  final tmp = Float32List(4);

  final real = QuantizedTensor.matVecBodyWGSL(GgmlType.q8_0,
      threadVar: 'trd', stride: '64u');
  final wOnly = real
      .replaceAll('* x[xi]', '')
      .replaceAll('* x[xi + 1u]', '')
      .replaceAll('* x[xi + 2u]', '')
      .replaceAll('* x[xi + 3u]', '');

  final variants = [
    ('A real   ', wrap(real), false),
    ('F w-only ', wrap(wOnly), false),
    ('C vec4x  ', wrap(bodyVec4x, vec4x: true), true),
    ('D unpack ', wrap(bodyUnpack, vec4x: true), true),
    ('E floor  ', wrap(bodyFloor), false),
    ('G no-mul ', wrap(bodyNoMul, vec4x: true), true),
    ('H no-ext ', wrap(bodyNoExtract, vec4x: true), true),
    (
      'J shared ',
      wrap(bodyShared, vec4x: true, shared: true, preBody: preShared),
      true
    ),
  ];

  for (final (label, src, vec4x) in variants) {
    try {
      final s = gpu.createComputeShader();
      s.loadKernelString(src);
      s.setBufferFire('wq', wq);
      s.setBufferFire(vec4x ? 'x4' : 'x', x);
      s.setBufferFire('y', y);
      s.dispatchFire(512, 1, 1);
      await y.read(tmp, 4);
      double bestUs = 1e18;
      for (int rep = 0; rep < reps; rep++) {
        final sw = Stopwatch()..start();
        for (int i = 0; i < chainN; i++) {
          s.dispatchFire(512, 1, 1);
        }
        await y.read(tmp, 4);
        final us = sw.elapsedMicroseconds / chainN;
        if (us < bestUs) bestUs = us;
      }
      s.destroy();
      final gbps = wBytes / (bestUs / 1e6) / 1e9;
      stdout.writeln('$label ${bestUs.toStringAsFixed(2)} us/dispatch  '
          '${gbps.toStringAsFixed(0)} GB/s (weight bytes)');
    } catch (e) {
      stdout.writeln('$label FAILED: ${e.toString().split('\n').first}');
    }
  }

  wq.destroy();
  x.destroy();
  y.destroy();
  exit(0);
}

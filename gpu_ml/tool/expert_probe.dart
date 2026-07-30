/// PRODUCTION-SCALE expert-kernel probe.
///
/// The in-process category map says the routed experts cost ~5.9 ms/token
/// for ~1.02 GB of q8_0 weights = ~173 GB/s, but tool/xs_shape_probe.dart
/// measured the SAME kernel shapes at 829-963 GB/s.  The probe differed in
/// one suspicious way: its whole weight buffer was ~25 MB (8 experts), which
/// FITS IN THE 4090's 72 MB L2 and was re-read 400 times — so it may have
/// been measuring L2, not DRAM.
///
/// This reproduces the real geometry — a 256-expert stack (272 MB per
/// tensor), 8 scattered expert ids, one token's worth of dispatches — and
/// varies ONE thing at a time:
///   A ids scattered over all 256 experts   (production)
///   B ids 0..7 (contiguous 8-expert window, = the old probe)
///   C ids all 0 (single 1 MB expert, fully L2-resident)
/// If A is much slower than C, the kernels are DRAM/footprint-bound and the
/// old probe's numbers were L2 fantasy.
///
///   dart run tool/expert_probe.dart
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:gpu_ml/gpu_ml.dart';
import 'package:minigpu/minigpu.dart';

const experts = 256, topK = 8;
const interRows = 512, dim = 2048;
const blocksPerToken = 40;
const reps = 3;

int bpe(int rows, int cols) => rows * cols ~/ 32 * 34;

String gateUpKernel(int t) {
  final r = 256 ~/ t;
  final accU = QuantizedTensor.accessorsWGSL
      .replaceAll('wq', 'wqu')
      .replaceAll('f16At', 'uf16At')
      .replaceAll('byteAt', 'ubyteAt')
      .replaceAll('wAt', 'uwAt');
  final bodyG = QuantizedTensor.sharedXBody(QuantizedTensor.matVecBodyWGSL(
      GgmlType.q8_0,
      threadVar: 'trd',
      stride: '${t}u'));
  final bodyU =
      bodyG.replaceAll('wq[', 'wqu[').replaceAll('f16At(', 'uf16At(');
  return '''
@group(0) @binding(0) var<storage, read_write> wq: array<u32>;
@group(0) @binding(1) var<storage, read_write> wqu: array<u32>;
@group(0) @binding(2) var<storage, read_write> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read_write> idxb: array<u32>;

const ROWS: u32 = ${interRows}u;
const COLS: u32 = ${dim}u;
const TKR: u32 = ${topK * interRows}u;

${QuantizedTensor.accessorsWGSL}
$accU
${QuantizedTensor.xsDeclWGSL(dim)}

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let slot: u32 = wid.z >> 1u;
  let gu: u32 = wid.z & 1u;
  let trd: u32 = lid.x % ${t}u;
  let row: u32 = wid.x * ${r}u + lid.x / ${t}u;
  // wid.y = block index: each block picks a DIFFERENT expert set, so one
  // dispatch touches a whole token's ~1 GB with no reuse (the L2 trap that
  // inflated the earlier probes).
  let eb: u32 = idxb[wid.y * ${topK}u + slot] * ${bpe(interRows, dim)}u;
  var acc: f32 = 0.0;
${QuantizedTensor.stageXsWGSL(dim)}
  if (row < ROWS) {
    if (gu == 0u) {
$bodyG
    } else {
$bodyU
    }
  }
  scratch[lid.x] = acc;
  workgroupBarrier();
  for (var s: u32 = ${t ~/ 2}u; s > 0u; s = s >> 1u) {
    if (trd < s) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s]; }
    workgroupBarrier();
  }
  if (trd == 0u && row < ROWS) {
    y[gu * TKR + slot * ROWS + row] = scratch[lid.x];
  }
}
''';
}

String downKernel(int t) {
  final r = 256 ~/ t;
  final body = QuantizedTensor.sharedXBody(QuantizedTensor.matVecBodyWGSL(
      GgmlType.q8_0,
      threadVar: 'trd',
      stride: '${t}u'));
  return '''
@group(0) @binding(0) var<storage, read_write> wq: array<u32>;
@group(0) @binding(1) var<storage, read_write> gu: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> idxb: array<u32>;

const ROWS: u32 = ${dim}u;
const COLS: u32 = ${interRows}u;
const C4: u32 = ${interRows ~/ 4}u;
const TKC4: u32 = ${topK * interRows ~/ 4}u;

${QuantizedTensor.accessorsWGSL}
${QuantizedTensor.xsDeclWGSL(interRows)}

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let slot: u32 = wid.z;
  let trd: u32 = lid.x % ${t}u;
  let row: u32 = wid.x * ${r}u + lid.x / ${t}u;
  let eb: u32 = idxb[wid.y * ${topK}u + slot] * ${bpe(dim, interRows)}u;
  var acc: f32 = 0.0;
  for (var i4x: u32 = lid.x; i4x < C4; i4x = i4x + 256u) {
    let gv: vec4<f32> = gu[slot * C4 + i4x];
    let uv: vec4<f32> = gu[TKC4 + slot * C4 + i4x];
    let v4x: vec4<f32> = (gv / (1.0 + exp(-gv))) * uv;
    let p4x: u32 = i4x * 4u + ((i4x * 4u) >> 5u);
    xs[p4x] = v4x.x; xs[p4x + 1u] = v4x.y;
    xs[p4x + 2u] = v4x.z; xs[p4x + 3u] = v4x.w;
  }
  workgroupBarrier();
  if (row < ROWS) {
$body
  }
  scratch[lid.x] = acc;
  workgroupBarrier();
  for (var s: u32 = ${t ~/ 2}u; s > 0u; s = s >> 1u) {
    if (trd < s) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s]; }
    workgroupBarrier();
  }
  if (trd == 0u && row < ROWS) { y[slot * ROWS + row] = scratch[lid.x]; }
}
''';
}

Future<void> main() async {
  final gpu = Minigpu.forAdapter('4090');
  await gpu.init();
  stdout.writeln('adapter: ${gpu.adapterName}');

  final gateBytes = experts * bpe(interRows, dim);
  final downBytes = experts * bpe(dim, interRows);
  stdout.writeln('stacks: gate/up ${(gateBytes >> 20)} MB each, '
      'down ${(downBytes >> 20)} MB');

  final wg = gpu.createBuffer(gateBytes, BufferDataType.uint32);
  final wu = gpu.createBuffer(gateBytes, BufferDataType.uint32);
  final wd = gpu.createBuffer(downBytes, BufferDataType.uint32);
  final x = gpu.createBuffer(dim * 4, BufferDataType.float32);
  final guOut = gpu.createBuffer(2 * topK * interRows * 4, BufferDataType.float32);
  final yOut = gpu.createBuffer(topK * dim * 4, BufferDataType.float32);
  final idxb =
      gpu.createBuffer(blocksPerToken * topK * 4, BufferDataType.uint32);
  final tmp = Float32List(4);

  const t = 16;
  final sGU = gpu.createComputeShader()..loadKernelString(gateUpKernel(t));
  sGU.setBufferFire('wq', wg);
  sGU.setBufferFire('wqu', wu);
  sGU.setBufferFire('x', x);
  sGU.setBufferFire('y', guOut);
  sGU.setBufferFire('idxb', idxb);
  final sD = gpu.createComputeShader()..loadKernelString(downKernel(t));
  sD.setBufferFire('wq', wd);
  sD.setBufferFire('gu', guOut);
  sD.setBufferFire('y', yOut);
  sD.setBufferFire('idxb', idxb);

  final guWgs = (interRows + (256 ~/ t) - 1) ~/ (256 ~/ t);
  final dWgs = (dim + (256 ~/ t) - 1) ~/ (256 ~/ t);
  // Bytes one token's worth of dispatches actually reads.
  final guBytes = blocksPerToken * topK * 2 * bpe(interRows, dim);
  final dBytes = blocksPerToken * topK * bpe(dim, interRows);

  Future<void> run(String label, List<int> ids) async {
    await idxb.write(Uint32List.fromList(ids), ids.length,
        dataType: BufferDataType.uint32);
    // warm
    sGU.dispatchFire(guWgs, blocksPerToken, topK * 2);
    sD.dispatchFire(dWgs, blocksPerToken, topK);
    await yOut.read(tmp, 4);

    double bestGu = 1e18, bestD = 1e18;
    for (int rep = 0; rep < reps; rep++) {
      var sw = Stopwatch()..start();
      sGU.dispatchFire(guWgs, blocksPerToken, topK * 2);
      await guOut.read(tmp, 4);
      final gu = sw.elapsedMicroseconds / 1000.0;
      if (gu < bestGu) bestGu = gu;

      sw = Stopwatch()..start();
      sD.dispatchFire(dWgs, blocksPerToken, topK);
      await yOut.read(tmp, 4);
      final d = sw.elapsedMicroseconds / 1000.0;
      if (d < bestD) bestD = d;
    }
    stdout.writeln('$label  gate+up ${bestGu.toStringAsFixed(2)} ms '
        '(${(guBytes / (bestGu / 1000) / 1e9).toStringAsFixed(0)} GB/s)   '
        'down ${bestD.toStringAsFixed(2)} ms '
        '(${(dBytes / (bestD / 1000) / 1e9).toStringAsFixed(0)} GB/s)   '
        'total ${(bestGu + bestD).toStringAsFixed(2)} ms/token');
  }

  // FOOTPRINT / BUFFER-SWITCH test: the model keeps ONE 816 MB stack per
  // block (40 buffers, ~32 GB total) and binds a different one for every
  // dispatch; this probe so far used a single 272 MB buffer for all 40.
  // Those are the only two differences left between "1.7 ms in the probe"
  // and "~6 ms in the model", and both are testable here for free.
  final extraStacks = int.parse(
      Platform.environment['PROBE_STACKS'] ?? '0'); // extra gate/up/down sets
  final stacksG = <Buffer>[wg], stacksU = <Buffer>[wu], stacksD = <Buffer>[wd];
  for (int i = 0; i < extraStacks; i++) {
    try {
      stacksG.add(gpu.createBuffer(gateBytes, BufferDataType.uint32));
      stacksU.add(gpu.createBuffer(gateBytes, BufferDataType.uint32));
      stacksD.add(gpu.createBuffer(downBytes, BufferDataType.uint32));
    } catch (e) {
      stdout.writeln('stopped at ${stacksG.length} stacks: '
          '${e.toString().split('\n').first}');
      break;
    }
  }
  if (extraStacks > 0) {
    stdout.writeln('allocated ${stacksG.length} stacks = '
        '${(stacksG.length * (2 * gateBytes + downBytes)) >> 30} GB');
  }

  /// One token's expert work the way the MODEL issues it: 40 separate
  /// dispatches, each binding that block's own stack.
  Future<void> runPerBlock(String label, List<int> ids) async {
    await idxb.write(Uint32List.fromList(ids), ids.length,
        dataType: BufferDataType.uint32);
    double best = 1e18;
    for (int rep = 0; rep < reps + 1; rep++) {
      final sw = Stopwatch()..start();
      for (int blk = 0; blk < blocksPerToken; blk++) {
        // PROBE_PIN=1: keep every dispatch on stack 0 while the other
        // stacks stay ALLOCATED — separates "many buffers exist" (VRAM
        // footprint / residency) from "we bind a different one each time".
        final s = Platform.environment['PROBE_PIN'] == '1'
            ? 0
            : blk % stacksG.length;
        sGU.setBufferFire('wq', stacksG[s]);
        sGU.setBufferFire('wqu', stacksU[s]);
        sGU.dispatchFire(guWgs, 1, topK * 2);
        sD.setBufferFire('wq', stacksD[s]);
        sD.dispatchFire(dWgs, 1, topK);
      }
      await yOut.read(tmp, 4);
      final ms = sw.elapsedMicroseconds / 1000.0;
      if (rep > 0 && ms < best) best = ms;
    }
    final bytes = guBytes + dBytes;
    stdout.writeln('$label  ${best.toStringAsFixed(2)} ms/token '
        '(${(bytes / (best / 1000) / 1e9).toStringAsFixed(0)} GB/s)');
    // Restore stack 0 bindings for later cases.
    sGU.setBufferFire('wq', wg);
    sGU.setBufferFire('wqu', wu);
    sD.setBufferFire('wq', wd);
  }

  final n = blocksPerToken * topK;
  final rng = math.Random(7);
  // A: production — every block routes to its own scattered expert set, so
  // one token touches ~1 GB of DISTINCT weights (no L2 reuse).
  await run('A unique/block (1 GB, no reuse)'.padRight(34),
      List<int>.generate(n, (_) => rng.nextInt(experts)));
  // B: same 8 scattered experts for all blocks (~25 MB, L2-resident) — the
  // shape the earlier probes accidentally measured.
  final eight = List<int>.generate(topK, (_) => rng.nextInt(experts));
  await run('B same 8 for all blocks (L2)'.padRight(34),
      List<int>.generate(n, (i) => eight[i % topK]));
  // C: one expert everywhere (~1 MB, entirely L2).
  await run('C single expert (1 MB, L2)'.padRight(34), List.filled(n, 0));

  // D: model-shaped issue — 40 separate dispatches, per-block stack binding,
  // unique experts.  Compare against A (same bytes, ONE fused dispatch) to
  // separate per-dispatch cost from bandwidth, and re-run with
  // PROBE_STACKS>0 to add the model's multi-GB footprint.
  final unique = List<int>.generate(n, (_) => rng.nextInt(experts));
  await runPerBlock(
      'D per-block dispatch x${stacksG.length} stack'.padRight(34), unique);

  for (final b in [wg, wu, wd, x, guOut, yOut, idxb]) {
    b.destroy();
  }
  exit(0);
}

/// x-lane-stride sweep: the wave-15 discriminator experiment.
///
/// Every variant reads the SAME total bytes from the same 512 MiB buffer with
/// the same per-warp footprint; the ONLY difference is how far apart the 32
/// lanes' addresses are at each load instruction (the per-lane "run length" R).
///
///   R=1  -> warp-cooperative: lanes read 32 consecutive words (1 line/instr)
///   R=8  -> each lane owns a 32 B run  (~q8_0 decode GEMV pattern, ~8 lines)
///   R=32 -> each lane owns a 128 B run (32 lines/instr, worst case)
///
/// If the L1-address-divergence model is right, useful GB/s tracks
/// 1/lines-per-warp-instruction and R=8..32 lands near the observed decode
/// GEMV ceiling (~230-330 GB/s) while R=1 approaches the streaming peak.
/// A flat curve kills the model (and vec4-x with it).
///
///   dart run tool/stride_probe.dart
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:minigpu/minigpu.dart';

const wgs = 1024; // workgroups (256 threads each) -> 8192 warps
const itScalar = 2048; // word-reads per thread  -> 2 GiB useful / dispatch
const itVec4 = 512; // vec4-reads per thread -> 2 GiB useful / dispatch
const nWords = 1 << 27; // 512 MiB buffer (>> 72 MB L2)
const reps = 3, dispatchesPerRep = 6;

String scalarKernel(int r) => '''
@group(0) @binding(0) var<storage, read_write> xb: array<f32>;
@group(0) @binding(1) var<storage, read_write> yb: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) lidx: u32) {
  let lane = lidx % 32u;
  let warpG = wid.x * 8u + lidx / 32u;
  let base = warpG * ${itScalar}u * 32u;
  var acc = 0.0;
  for (var i = 0u; i < ${itScalar}u; i = i + 1u) {
    let j = i / ${r}u;
    let rr = i % ${r}u;
    let addr = (base + j * ${32 * r}u + lane * ${r}u + rr) & ${nWords - 1}u;
    acc = acc + xb[addr];
  }
  yb[wid.x * 256u + lidx] = acc;
}
''';

String vec4Kernel(int r4) => '''
@group(0) @binding(0) var<storage, read_write> xb: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> yb: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) lidx: u32) {
  let lane = lidx % 32u;
  let warpG = wid.x * 8u + lidx / 32u;
  let base = warpG * ${itVec4}u * 32u;
  var acc = vec4<f32>(0.0);
  for (var i = 0u; i < ${itVec4}u; i = i + 1u) {
    let j = i / ${r4}u;
    let rr = i % ${r4}u;
    let addr = (base + j * ${32 * r4}u + lane * ${r4}u + rr) & ${(nWords >> 2) - 1}u;
    acc = acc + xb[addr];
  }
  yb[wid.x * 256u + lidx] = acc.x + acc.y + acc.z + acc.w;
}
''';

Future<void> main() async {
  final gpu = Minigpu.forAdapter('4090');
  await gpu.init();
  stdout.writeln('adapter: ${gpu.adapterName}');

  final xb = gpu.createBuffer(nWords * 4, BufferDataType.float32);
  final yb = gpu.createBuffer(wgs * 256 * 4, BufferDataType.float32);
  final tmp = Float32List(4);
  const usefulPerDispatch = wgs * 256 * itScalar * 4; // == vec4 variant too

  Future<double> bench(String src) async {
    final s = gpu.createComputeShader();
    s.loadKernelString(src);
    s.setBuffer('xb', xb);
    s.setBuffer('yb', yb);
    s.dispatchFire(wgs, 1, 1); // pipeline compile + warmup
    await yb.read(tmp, 4);
    double best = 0;
    for (int rep = 0; rep < reps; rep++) {
      final sw = Stopwatch()..start();
      for (int d = 0; d < dispatchesPerRep; d++) {
        s.dispatchFire(wgs, 1, 1);
      }
      await yb.read(tmp, 4);
      final gbps = usefulPerDispatch *
          dispatchesPerRep /
          (sw.elapsedMicroseconds / 1e6) /
          1e9;
      if (gbps > best) best = gbps;
    }
    s.destroy();
    return best;
  }

  stdout.writeln('--- scalar f32 loads (lane run R words) ---');
  for (final r in [1, 2, 4, 8, 16, 32]) {
    final gbps = await bench(scalarKernel(r));
    stdout.writeln('R=$r'.padRight(6) +
        'laneStride=${(r * 4).toString().padLeft(3)}B  '
        'lines/warp-instr~${r.clamp(1, 32).toString().padLeft(2)}  '
        '${gbps.toStringAsFixed(1)} GB/s');
  }

  stdout.writeln('--- vec4 f32 loads (lane run R4 vec4s) ---');
  for (final r4 in [1, 2, 8]) {
    final gbps = await bench(vec4Kernel(r4));
    stdout.writeln('R4=$r4'.padRight(6) +
        'laneStride=${(r4 * 16).toString().padLeft(3)}B  '
        '${gbps.toStringAsFixed(1)} GB/s');
  }

  xb.destroy();
  yb.destroy();
  exit(0);
}

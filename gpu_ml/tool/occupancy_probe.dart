/// Occupancy + dispatch-overhead probes: the two remaining suspects for the
/// decode GEMV wall after the stride sweep came back flat (~1.0-1.1 TB/s at
/// every lane stride, full occupancy).
///
/// 1. GB/s vs workgroup count with the decode-like access pattern (R=8):
///    a real 2048-row GEMV launches ~8 workgroups; the probe that hit
///    1.1 TB/s launched 1024.  This measures the bandwidth actually
///    available at GEMV-realistic occupancy.
/// 2. Per-dispatch overhead: N back-to-back fire-and-forget dispatches of a
///    trivial kernel through the same batching path decode uses.  Decode
///    issues hundreds of barriered dispatches per token; if each costs
///    20-40 us the bubbles alone explain most of 28 ms/token.
///
///   dart run tool/occupancy_probe.dart
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:minigpu/minigpu.dart';

const nWords = 1 << 27; // 512 MiB
const reps = 3;

String streamKernel(int it) => '''
@group(0) @binding(0) var<storage, read_write> xb: array<f32>;
@group(0) @binding(1) var<storage, read_write> yb: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) lidx: u32) {
  let lane = lidx % 32u;
  let warpG = wid.x * 8u + lidx / 32u;
  let base = warpG * ${it}u * 32u;
  var acc = 0.0;
  for (var i = 0u; i < ${it}u; i = i + 1u) {
    let j = i / 8u;
    let rr = i % 8u;
    let addr = (base + j * 256u + lane * 8u + rr) & ${nWords - 1}u;
    acc = acc + xb[addr];
  }
  yb[wid.x * 256u + lidx] = acc;
}
''';

const tinyKernel = '''
@group(0) @binding(0) var<storage, read_write> xb: array<f32>;
@group(0) @binding(1) var<storage, read_write> yb: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) lidx: u32) {
  var acc = 0.0;
  for (var i = 0u; i < 32u; i = i + 1u) {
    acc = acc + xb[(lidx * 32u + i) & ${nWords - 1}u];
  }
  yb[lidx] = acc;
}
''';

Future<void> main() async {
  final gpu = Minigpu.forAdapter('4090');
  await gpu.init();
  stdout.writeln('adapter: ${gpu.adapterName}');

  final xb = gpu.createBuffer(nWords * 4, BufferDataType.float32);
  final yb = gpu.createBuffer(1024 * 256 * 4, BufferDataType.float32);
  final tmp = Float32List(4);

  stdout.writeln('--- GB/s vs workgroups (R=8 decode-like pattern) ---');
  for (final wgs in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]) {
    // Hold useful bytes per dispatch at 1 GiB regardless of occupancy.
    final it = (1 << 28) ~/ (wgs * 256);
    final s = gpu.createComputeShader();
    s.loadKernelString(streamKernel(it));
    s.setBufferFire('xb', xb);
    s.setBufferFire('yb', yb);
    s.dispatchFire(wgs, 1, 1);
    await yb.read(tmp, 4);
    double best = 0;
    for (int rep = 0; rep < reps; rep++) {
      final sw = Stopwatch()..start();
      s.dispatchFire(wgs, 1, 1);
      s.dispatchFire(wgs, 1, 1);
      await yb.read(tmp, 4);
      final gbps = 2 * (1 << 28) * 4 / (sw.elapsedMicroseconds / 1e6) / 1e9;
      if (gbps > best) best = gbps;
    }
    s.destroy();
    stdout.writeln('wgs=${wgs.toString().padLeft(4)}  '
        'threads=${(wgs * 256).toString().padLeft(6)}  '
        '${best.toStringAsFixed(1)} GB/s');
  }

  stdout.writeln('--- per-dispatch overhead (fire-and-forget batch) ---');
  final s = gpu.createComputeShader();
  s.loadKernelString(tinyKernel);
  s.setBufferFire('xb', xb);
  s.setBufferFire('yb', yb);
  s.dispatchFire(1, 1, 1);
  await yb.read(tmp, 4);
  for (final n in [64, 256, 512]) {
    double bestUs = 1e18;
    for (int rep = 0; rep < reps; rep++) {
      final sw = Stopwatch()..start();
      for (int i = 0; i < n; i++) {
        s.dispatchFire(1, 1, 1);
      }
      await yb.read(tmp, 4);
      final us = sw.elapsedMicroseconds / n;
      if (us < bestUs) bestUs = us;
    }
    stdout.writeln('n=$n'.padRight(7) +
        '${bestUs.toStringAsFixed(2)} us/dispatch');
  }
  s.destroy();

  xb.destroy();
  yb.destroy();
  exit(0);
}

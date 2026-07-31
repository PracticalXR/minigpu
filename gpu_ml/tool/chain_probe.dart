/// Dependent-chain dispatch probe: decode reality is ~800 serially-dependent
/// dispatches per token across many pipelines.  The batched same-pipeline
/// probe measured 5.85 us/dispatch, but real decode averages ~28 us.  This
/// measures a chain that mimics decode: two pipelines ping-ponging between
/// two buffers (true RAW dependency + pipeline switch every dispatch), at
/// three kernel sizes (tiny, dense-qmv-like 4 MB, expert-like 9 MB).
///
/// Run with MGPU_BACKEND=d3d12 / vulkan to compare Dawn backends.
///
///   dart run tool/chain_probe.dart
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:minigpu/minigpu.dart';

const reps = 3;

/// Kernel reading [words] f32s spread over [wgs] workgroups from src,
/// writing one value per thread to dst — direction baked per pipeline.
String chainKernel(String src, String dst, int wgs, int itPerThread) => '''
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) lidx: u32) {
  let tid = wid.x * 256u + lidx;
  var acc = 0.0;
  for (var i = 0u; i < ${itPerThread}u; i = i + 1u) {
    acc = acc + $src[(tid * ${itPerThread}u + i) & ${(1 << 20) - 1}u];
  }
  $dst[tid & ${(1 << 20) - 1}u] = acc + f32(tid % 2u);
}
''';

Future<void> main() async {
  final gpu = Minigpu.forAdapter('4090');
  await gpu.init();
  stdout.writeln('adapter: ${gpu.adapterName} '
      'backend=${Platform.environment['MGPU_BACKEND'] ?? 'default(d3d11)'}');

  final bufA = gpu.createBuffer((1 << 20) * 4, BufferDataType.float32);
  final bufB = gpu.createBuffer((1 << 20) * 4, BufferDataType.float32);
  final tmp = Float32List(4);

  // (label, workgroups, f32 reads per thread) — sized like real decode kernels.
  final configs = [
    ('tiny (1 wg)', 1, 32),
    ('qmv-like (512 wg, 4 MB)', 512, 8),
    ('expert-like (2048 wg, 9 MB)', 2048, 4),
  ];

  for (final (label, wgs, it) in configs) {
    final sAB = gpu.createComputeShader();
    sAB.loadKernelString(chainKernel('a', 'b', wgs, it));
    sAB.setBuffer('a', bufA);
    sAB.setBuffer('b', bufB);
    final sBA = gpu.createComputeShader();
    sBA.loadKernelString(chainKernel('b', 'a', wgs, it));
    sBA.setBuffer('a', bufA);
    sBA.setBuffer('b', bufB);
    sAB.dispatchFire(wgs, 1, 1);
    sBA.dispatchFire(wgs, 1, 1);
    await bufA.read(tmp, 4);

    const n = 400; // 200 ping-pong pairs ~ one decode token's chain length
    double bestUs = 1e18;
    for (int rep = 0; rep < reps; rep++) {
      final sw = Stopwatch()..start();
      for (int i = 0; i < n ~/ 2; i++) {
        sAB.dispatchFire(wgs, 1, 1);
        sBA.dispatchFire(wgs, 1, 1);
      }
      await bufA.read(tmp, 4);
      final us = sw.elapsedMicroseconds / n;
      if (us < bestUs) bestUs = us;
    }
    stdout.writeln('$label: ${bestUs.toStringAsFixed(2)} us/dispatch '
        '(chain of $n -> ${(bestUs * 800 / 1000).toStringAsFixed(1)} ms per '
        '800-dispatch token)');

    // REBIND variant: decode re-fires setBufferFire before every dispatch
    // (~2400 binds/token) — measure the per-dispatch cost with the real
    // bind cadence.
    double rebindUs = 1e18;
    for (int rep = 0; rep < reps; rep++) {
      final sw = Stopwatch()..start();
      for (int i = 0; i < n ~/ 2; i++) {
        sAB.setBuffer('a', bufA);
        sAB.setBuffer('b', bufB);
        sAB.dispatchFire(wgs, 1, 1);
        sBA.setBuffer('a', bufA);
        sBA.setBuffer('b', bufB);
        sBA.dispatchFire(wgs, 1, 1);
      }
      await bufA.read(tmp, 4);
      final us = sw.elapsedMicroseconds / n;
      if (us < rebindUs) rebindUs = us;
    }
    stdout.writeln('$label REBIND: ${rebindUs.toStringAsFixed(2)} us/dispatch '
        '(-> ${(rebindUs * 800 / 1000).toStringAsFixed(1)} ms per '
        '800-dispatch token)');
    sAB.destroy();
    sBA.destroy();
  }

  // INDEPENDENT-KERNEL OVERLAP TEST: two kernels with fully disjoint
  // buffers, each sized so solo exec is measurable (~10 us).  If running
  // them interleaved costs sum(solo) the backend serializes ALL dispatches
  // (D3D11 implicit UAV sync); if ~max(solo) they overlap.
  final bufC = gpu.createBuffer((1 << 22) * 4, BufferDataType.float32);
  final bufD = gpu.createBuffer((1 << 20) * 4, BufferDataType.float32);
  final bufE = gpu.createBuffer((1 << 22) * 4, BufferDataType.float32);
  final bufF = gpu.createBuffer((1 << 20) * 4, BufferDataType.float32);
  String soloKernel(int wgs, int it) => '''
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) lidx: u32) {
  let tid = wid.x * 256u + lidx;
  var acc = 0.0;
  for (var i = 0u; i < ${it}u; i = i + 1u) {
    acc = acc + a[(tid * ${it}u + i) & ${(1 << 22) - 1}u];
  }
  b[tid & ${(1 << 20) - 1}u] = acc;
}
''';
  final sC = gpu.createComputeShader();
  sC.loadKernelString(soloKernel(256, 64));
  sC.setBuffer('a', bufC);
  sC.setBuffer('b', bufD);
  final sE = gpu.createComputeShader();
  sE.loadKernelString(soloKernel(256, 64));
  sE.setBuffer('a', bufE);
  sE.setBuffer('b', bufF);
  sC.dispatchFire(256, 1, 1);
  sE.dispatchFire(256, 1, 1);
  await bufD.read(tmp, 4);

  Future<double> time(void Function() fire, int n) async {
    double best = 1e18;
    for (int rep = 0; rep < reps; rep++) {
      final sw = Stopwatch()..start();
      fire();
      await bufD.read(tmp, 4);
      final us = sw.elapsedMicroseconds / n;
      if (us < best) best = us;
    }
    return best;
  }

  final soloC = await time(() {
    for (int i = 0; i < 200; i++) {
      sC.dispatchFire(256, 1, 1);
    }
  }, 200);
  final soloE = await time(() {
    for (int i = 0; i < 200; i++) {
      sE.dispatchFire(256, 1, 1);
    }
  }, 200);
  final inter = await time(() {
    for (int i = 0; i < 200; i++) {
      sC.dispatchFire(256, 1, 1);
      sE.dispatchFire(256, 1, 1);
    }
  }, 200); // us per PAIR
  stdout.writeln('overlap: soloC=${soloC.toStringAsFixed(2)} us '
      'soloE=${soloE.toStringAsFixed(2)} us pair=${inter.toStringAsFixed(2)} us '
      '(sum=${(soloC + soloE).toStringAsFixed(2)}, serialized if pair~sum)');
  sC.destroy();
  sE.destroy();
  bufC.destroy();
  bufD.destroy();
  bufE.destroy();
  bufF.destroy();

  bufA.destroy();
  bufB.destroy();
  exit(0);
}

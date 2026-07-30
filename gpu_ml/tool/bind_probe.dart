/// Isolates the cost of binding a DIFFERENT buffer to a shader.
///
/// ComputeShader keeps exactly one WGPUBindGroup: setBuffer early-outs when
/// the buffer pointer is unchanged, but any change releases the bind group
/// and creates a new one on the next dispatch.  The expert probe implied
/// ~10 us per distinct bind, which — at ~680 per decode token — would be
/// half the token.  This measures it with a trivial kernel so no bandwidth
/// is involved, and varies buffer SIZE to see whether the cost is per-bind
/// or per-byte (which decides whether consolidating buffers can fix it).
///
///   dart run tool/bind_probe.dart
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:minigpu/minigpu.dart';

const n = 200; // dispatches per timed run
const reps = 5;

const kernel = '''
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lidx: u32) {
  b[lidx] = a[lidx] + 1.0;
}
''';

Future<void> main() async {
  final gpu = Minigpu.forAdapter('4090');
  await gpu.init();
  stdout.writeln('adapter: ${gpu.adapterName}');

  final out = gpu.createBuffer(1024, BufferDataType.float32);
  final tmp = Float32List(4);
  final s = gpu.createComputeShader()..loadKernelString(kernel);

  Future<double> timed(List<Buffer> pool) async {
    s.setBufferFire('a', pool[0]);
    s.setBufferFire('b', out);
    s.dispatchFire(1, 1, 1);
    await out.read(tmp, 4);
    double best = 1e18;
    for (int rep = 0; rep < reps; rep++) {
      final sw = Stopwatch()..start();
      for (int i = 0; i < n; i++) {
        s.setBufferFire('a', pool[i % pool.length]);
        s.dispatchFire(1, 1, 1);
      }
      await out.read(tmp, 4);
      final us = sw.elapsedMicroseconds / n;
      if (us < best) best = us;
    }
    return best;
  }

  Future<void> caseFor(String label, int count, int bytes) async {
    final pool = <Buffer>[];
    for (int i = 0; i < count; i++) {
      pool.add(gpu.createBuffer(bytes, BufferDataType.float32));
    }
    final us = await timed(pool);
    stdout.writeln('${label.padRight(38)}${us.toStringAsFixed(2)} us/dispatch'
        '   -> ${(us * 680 / 1000).toStringAsFixed(1)} ms per 680 binds');
    for (final b in pool) {
      b.destroy();
    }
  }

  // Baseline: one buffer, rebound every dispatch (setBuffer early-outs).
  await caseFor('same 1 MB buffer (no rebind)', 1, 1 << 20);
  // Cycling distinct buffers, small then large: if the delta is flat in
  // size, the cost is per-BIND and consolidating buffers fixes it.
  await caseFor('cycle 40 x 1 MB', 40, 1 << 20);
  await caseFor('cycle 40 x 64 MB', 40, 64 << 20);
  await caseFor('cycle 40 x 272 MB', 40, 272 << 20);
  await caseFor('cycle 8 x 272 MB', 8, 272 << 20);
  await caseFor('cycle 2 x 272 MB', 2, 272 << 20);

  s.destroy();
  out.destroy();
  exit(0);
}

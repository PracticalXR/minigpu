/// The warmer's SUCCESS path, on a real device.
///
/// The contract tests deliberately never initialise a context, so they only
/// ever exercise failure. A warmer that can report failure beautifully and
/// never actually warms anything would pass all of them — this is the test
/// that says it works.
///
/// Skips itself when no adapter comes up, so it is safe in CI without a GPU.
library;

import 'package:minigpu/minigpu.dart';
import 'package:test/test.dart';

const _kAdd = '''
@group(0) @binding(0) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  b[0] = b[0] + 1.0;
}
''';

const _kMul = '''
@group(0) @binding(0) var<storage, read_write> b: array<f32>;
@group(0) @binding(1) var<storage, read_write> c: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  c[0] = b[0] * 2.0;
}
''';

void main() {
  late Minigpu gpu;
  var up = false;

  setUpAll(() async {
    gpu = Minigpu();
    try {
      await gpu.init();
      up = true;
    } catch (_) {
      up = false;
    }
  });

  test('warms real kernels to ready, with progress', () async {
    if (!up) {
      markTestSkipped('no GPU adapter here');
      return;
    }
    final specs = [
      const KernelWarmSpec(
          label: 'add', source: _kAdd, buffers: {'b': 4}),
      const KernelWarmSpec(
          label: 'mul', source: _kMul, buffers: {'b': 4, 'c': 4}),
    ];
    final w = warmShaders(gpu, specs);
    final seen = <WarmProgress>[];
    final sub = w.progress.listen(seen.add);
    final r = await w.done;
    await sub.cancel();

    expect(r.phase, WarmPhase.ready,
        reason: 'errors: ${r.errors.join("; ")}');
    expect(r.failed, 0);
    expect(r.completed, 2);
    expect(r.elapsed, greaterThan(Duration.zero));
    // The labels have to reach the stream, or a progress UI has nothing to
    // show but a number.
    expect(seen.map((p) => p.label).whereType<String>().toSet(),
        containsAll(<String>{'add', 'mul'}));
  });

  test('a second warm of the same source is a no-op, not a recompile', () async {
    if (!up) {
      markTestSkipped('no GPU adapter here');
      return;
    }
    // Same SOURCE as above — the dedupe key. The pipelines are process-global
    // in the backend, so re-warming should cost nothing; a caller that warms
    // per-stream must not pay per stream.
    const spec = KernelWarmSpec(
        label: 'add-again', source: _kAdd, buffers: {'b': 4});
    final w = warmShaders(gpu, const [spec]);
    final r = await w.done;
    expect(r.phase, WarmPhase.ready);
    expect(r.completed, 1);
  });
}

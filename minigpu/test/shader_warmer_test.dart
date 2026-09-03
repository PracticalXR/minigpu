/// The warmer's CONTRACT, tested without a GPU.
///
/// Everything asserted here is a rule that exists because breaking it has hurt
/// before: a progress bar that goes backwards, a startup future that throws
/// into nothing, a dead device costing one timeout per remaining kernel, a
/// panel that subscribes late and renders an empty bar.
///
/// These use a Minigpu that was deliberately never `init()`ed, so every
/// dispatch fails — which is exactly the interesting half, and it is
/// deterministic on a machine that HAS a GPU as well as one that does not.
/// The success path needs a live context and is covered by
/// `shader_warmer_gpu_test.dart`.
library;

import 'package:minigpu/minigpu.dart';
import 'package:test/test.dart';

const _kTrivial = '''
@group(0) @binding(0) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  b[0] = b[0] + 1.0;
}
''';

KernelWarmSpec _spec(String label, {Duration? timeout}) => KernelWarmSpec(
      label: label,
      // Distinct source per label: the warmer DEDUPES by source text, so
      // reusing one string would make later specs no-op and quietly weaken
      // every count below.
      source: '// $label\n$_kTrivial',
      buffers: const {'b': 4},
      timeout: timeout ?? const Duration(seconds: 2),
    );

void main() {
  group('ShaderWarmer contract', () {
    test('an empty set completes ready, not stuck', () async {
      final w = warmShaders(Minigpu(), const []);
      final r = await w.done;
      expect(r.phase, WarmPhase.ready);
      expect(r.total, 0);
      expect(r.fraction, 1.0, reason: 'an empty bar is a FULL bar, not 0/0');
    });

    test('done NEVER throws, even when every kernel fails', () async {
      final w = warmShaders(Minigpu(), [_spec('a'), _spec('b')]);
      // The assertion is that awaiting does not throw. Without a GPU these
      // all fail, which is the case most likely to reject.
      final r = await w.done;
      expect(r.isDone, isTrue);
      expect(r.total, 2);
      expect(r.completed, 2,
          reason: 'a FAILED kernel still counts as completed — a progress bar '
              'that stalls on failure is how a warm looks like a hang');
    });

    test('progress is monotonic and ends complete', () async {
      final w = warmShaders(Minigpu(), [_spec('a'), _spec('b'), _spec('c')]);
      final seen = <WarmProgress>[];
      final sub = w.progress.listen(seen.add);
      final r = await w.done;
      await sub.cancel();
      expect(seen, isNotEmpty);
      var last = -1;
      for (final p in seen) {
        expect(p.completed, greaterThanOrEqualTo(last),
            reason: 'completed went BACKWARDS: ${seen.map((s) => s.completed)}');
        expect(p.completed, lessThanOrEqualTo(p.total));
        last = p.completed;
      }
      expect(r.completed, r.total);
    });

    test('a late subscriber still sees current state (replay)', () async {
      final w = warmShaders(Minigpu(), [_spec('a')]);
      await w.done;
      // Subscribing AFTER everything finished must still yield a snapshot.
      final first = await w.progress.first;
      expect(first.isDone, isTrue,
          reason: 'a panel that subscribes late must not render an empty bar');
      expect(w.value.isDone, isTrue);
    });

    test('failures are named, with a label and a stage', () async {
      final w = warmShaders(Minigpu(), [_spec('kernel-under-test')]);
      final r = await w.done;
      expect(r.phase, WarmPhase.failed);
      expect(r.errors, hasLength(1));
      expect(r.errors.single.label, 'kernel-under-test',
          reason: 'an error nobody can attribute to a kernel is not a report');
      expect(r.errors.single.stage, isA<WarmStage>());
      expect(r.errors.single.message, isNotEmpty);
      expect(r.failed, 1);
    });

    test('cancel stops the set and reports cancelled', () async {
      final w = warmShaders(
          Minigpu(), List.generate(8, (i) => _spec('k$i')));
      w.cancel();
      final r = await w.done;
      expect(r.phase, anyOf(WarmPhase.cancelled, WarmPhase.failed),
          reason: 'cancel may land after the last kernel, but must never hang');
      expect(r.isDone, isTrue);
    });

    test('a per-spec timeout is that spec, not the set', () async {
      final w = warmShaders(Minigpu(), [
        _spec('quick', timeout: const Duration(milliseconds: 50)),
        _spec('also-quick', timeout: const Duration(milliseconds: 50)),
      ]);
      final r = await w.done;
      expect(r.total, 2);
      expect(r.completed, 2,
          reason: 'the SECOND spec must still be attempted after the first '
              'fails — otherwise one big kernel hides every kernel after it');
    });

    test('WarmProgress.fraction tracks completed/total', () {
      const p = WarmProgress(
        phase: WarmPhase.running,
        completed: 1,
        total: 4,
        failed: 0,
        elapsed: Duration.zero,
      );
      expect(p.fraction, 0.25);
      expect(p.isDone, isFalse);
    });
  });
}

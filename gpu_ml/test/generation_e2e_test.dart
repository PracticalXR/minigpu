@TestOn('windows || mac-os || linux')
@Timeout(Duration(minutes: 30))
library;

import 'dart:io';

import 'package:gpu_ml/gpu_ml_io.dart';
import 'package:test/test.dart';

/// THE end-to-end milestone: load the real 40 GB Qwen3.6-35B-A3B GGUF and
/// greedily generate text.  Heavy (minutes): gated behind RUN_E2E=1 so the
/// regular suite stays fast.
///
///   RUN_E2E=1 dart test test/generation_e2e_test.dart --concurrency 1
const q8Path =
    r'C:\models\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf';

void main() {
  final enabled =
      Platform.environment['RUN_E2E'] == '1' && File(q8Path).existsSync();

  test('greedy generation from the real model produces sane text', () async {
    final sw = Stopwatch()..start();
    final model = await Qwen35Model.load(
      q8Path,
      onProgress: (m) {
        if (m == 'ready' || m.contains('blk.0 ') || m == 'loading lm_head') {
          print('[load ${sw.elapsed}] $m');
        }
      },
    );
    print('[load done ${sw.elapsed}] vocab=${model.vocab}');

    // Tokenizer sanity round-trip before burning GPU time.
    final ids = model.tokenizer.encode('The capital of France is');
    expect(ids, isNotEmpty);
    expect(model.tokenizer.decode(ids), equals('The capital of France is'));
    print('prompt tokens: $ids');

    final genSw = Stopwatch()..start();
    final out = StringBuffer();
    final tokens = await model.generate(
      'The capital of France is',
      maxTokens: 8,
      onToken: (id, text) {
        out.write(text);
        print('[${genSw.elapsed}] token $id -> "$text" '
            '(cache ${model.cacheHits}h/${model.cacheMisses}m)');
      },
    );

    final text = out.toString();
    print('GENERATED: "$text"');
    print('expert cache: ${model.cacheHits} hits / ${model.cacheMisses} misses');

    expect(tokens, isNotEmpty);
    expect(text.trim(), isNotEmpty);
    // A correctly wired 35B model completes this greedily with certainty;
    // garbage math will not.
    expect(text, contains('Paris'));

    await model.destroy();
  }, skip: enabled ? false : 'set RUN_E2E=1 (and have the model file) to run');
}

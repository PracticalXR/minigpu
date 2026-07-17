/// Regression tests for TensorShaderCache — tensor ops used to create,
/// compile and destroy a compute shader on EVERY call (per-frame Tint
/// compilation for anything invoking ops at frame rate, e.g. gpu_pipeline's
/// StreamMergeStage).  Ops now acquire cached shaders keyed by full WGSL
/// source; these tests pin the caching semantics and, critically, that
/// REUSED shaders still produce correct results with re-bound buffers.
library;

import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:gpu_tensor/src/gpu_helpers.dart';
import 'package:minigpu/minigpu.dart';

void main() {
  late Minigpu gpu;

  setUpAll(() async {
    gpu = Minigpu();
    if (!gpu.isInitialized) {
      await gpu.init();
    }
  });

  Future<Tensor> make(List<double> values) async {
    final t = await Tensor.create([values.length]);
    await t.write(Float32List.fromList(values));
    return t;
  }

  Future<List<double>> read(Tensor t) async {
    final data = await t.getData() as Float32List;
    return data.toList();
  }

  test('same op + size reuses one shader; results stay correct', () async {
    TensorShaderCache.clear();

    final a = await make([1, 2, 3, 4]);
    final b = await make([10, 20, 30, 40]);
    final c = await make([100, 200, 300, 400]);

    final r1 = await a.add(b);
    final afterFirst = TensorShaderCache.sizeFor(a.gpu);

    // Same op, same size, DIFFERENT buffers: must hit the cache and still
    // compute with the newly bound buffers, not the previous ones.
    final r2 = await b.add(c);
    final afterSecond = TensorShaderCache.sizeFor(a.gpu);

    expect(await read(r1), [11, 22, 33, 44]);
    expect(await read(r2), [110, 220, 330, 440]);
    expect(
      afterSecond,
      afterFirst,
      reason: 'second same-shape add must not compile a new shader',
    );
  });

  test('different sizes produce distinct cache entries', () async {
    TensorShaderCache.clear();

    final a4 = await make([1, 1, 1, 1]);
    final b4 = await make([2, 2, 2, 2]);
    await a4.add(b4);
    final afterSize4 = TensorShaderCache.sizeFor(a4.gpu);

    final a8 = await make([1, 1, 1, 1, 1, 1, 1, 1]);
    final b8 = await make([3, 3, 3, 3, 3, 3, 3, 3]);
    final r = await a8.add(b8);
    final afterSize8 = TensorShaderCache.sizeFor(a4.gpu);

    expect(await read(r), everyElement(4.0));
    expect(
      afterSize8,
      greaterThan(afterSize4),
      reason: 'a new size bakes a new source → new cache entry',
    );
  });

  test('interleaved distinct ops stay correct through the cache', () async {
    TensorShaderCache.clear();

    final a = await make([1, 2, 3, 4]);
    final b = await make([5, 1, 7, 2]);

    // max / min / multiplyScalar / addScalar exercise several cached
    // shaders in sequence, mirroring a merge + post-processing frame.
    expect(await read(await a.max(b)), [5, 2, 7, 4]);
    expect(await read(await a.min(b)), [1, 1, 3, 2]);
    expect(await read(await a.multiplyScalar(2.0)), [2, 4, 6, 8]);
    expect(await read(await a.addScalar(1.0)), [2, 3, 4, 5]);

    // Second round on the SAME shaders (all cache hits now).
    final before = TensorShaderCache.sizeFor(a.gpu);
    expect(await read(await b.max(a)), [5, 2, 7, 4]);
    expect(await read(await b.min(a)), [1, 1, 3, 2]);
    expect(TensorShaderCache.sizeFor(a.gpu), before);
  });

  test('repeated frame-like op churn does not grow the cache', () async {
    TensorShaderCache.clear();

    final a = await make([1, 2, 3, 4]);
    final b = await make([4, 3, 2, 1]);
    await a.max(b);
    final steady = TensorShaderCache.sizeFor(a.gpu);

    // 50 "frames" of the merge hot path: previously 50 shader compiles.
    for (var i = 0; i < 50; i++) {
      final r = await a.max(b);
      expect(await read(r), [4, 3, 3, 4]);
    }
    expect(TensorShaderCache.sizeFor(a.gpu), steady);
  });
}

import 'package:gpu_pipeline/gpu_pipeline.dart';
import 'package:gpu_pipeline/src/pipeline_resources.dart';
import 'package:test/test.dart';

const _mb = 1024 * 1024;

void main() {
  // Both used to be accepted and ignored: the cache was always 100 and 200.
  test('the cache sizes reach the resource cache', () async {
    final pipeline = Pipeline(
      id: 'sizes',
      maxCachedShaders: 7,
      maxCachedTensors: 9,
    );
    expect(pipeline.resourceCache.maxShaders, 7);
    expect(pipeline.resourceCache.maxTensors, 9);
    await pipeline.dispose();
  });

  test('a pipeline sets the limit of the asset manager it creates', () async {
    final byDefault = Pipeline(id: 'own');
    final small = Pipeline(id: 'own_small', maxMemoryMB: 8);
    expect(byDefault.assetManager.cacheLimit, 4096 * _mb);
    expect(small.assetManager.cacheLimit, 8 * _mb);
    await byDefault.dispose();
    await small.dispose();
  });

  // PipelineLoader and RuntimeBuilder hand one manager to every pipeline
  // they build. Each pipeline used to reset its limit to 4096 MB.
  test('a shared asset manager keeps its limit', () async {
    final shared = AssetManager();
    await shared.setCacheLimit(1024);

    final a = Pipeline(id: 'a', assetManager: shared);
    final b = Pipeline(id: 'b', assetManager: shared);
    expect(shared.cacheLimit, 1024);

    final c = Pipeline(id: 'c', assetManager: shared, maxMemoryMB: 2);
    expect(shared.cacheLimit, 2 * _mb, reason: 'an explicit limit applies');

    for (final p in [a, b, c]) {
      await p.dispose();
    }
    await shared.dispose();
  });

  // start() used to check these against two invented GPUs, and neither had
  // the hardware codec that mlInference() asks for, so start() threw.
  test('a stage with ML resource hints starts', () async {
    final pipeline = Pipeline(id: 'hints');
    pipeline.addStage(
      StageBuilder('half')
          // ignore: deprecated_member_use_from_same_package
          .withResourceRequirements(
            requirements: ResourceRequirements.mlInference(),
          )
          .op('multiplyScalar', arguments: [0.5])
          .build(),
    );

    await pipeline.start();
    expect(pipeline.isRunning, isTrue);
    await pipeline.stop();
    await pipeline.dispose();
  });
}

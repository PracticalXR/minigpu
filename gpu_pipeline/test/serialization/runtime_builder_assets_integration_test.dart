import 'dart:convert';
import 'package:test/test.dart';

import 'package:gpu_pipeline/src/serialization/runtime_builder.dart';
import 'package:gpu_pipeline/src/serialization/pipeline_models.dart';
import 'package:gpu_pipeline/src/pipeline_assets.dart';

void main() {
  group('RuntimeBuilder + serialized assets + Pipeline.start', () {
    test('NetworkConfig.assets are threaded and preloaded on start', () async {
      // Shared AssetManager injected by the builder
      final shared = AssetManager();
      final builder = RuntimeBuilder(sharedAssets: shared);

      // Network with one empty pipeline and two serialized assets
      final net = NetworkConfig(
        pipelines: [
          PipelineConfig(
            id: 'p1',
            stages: const [], // no stages required for this test
            connections: const [], // no connections
            queue: const {}, // no queue
          ),
        ],
        assets: [
          AssetConfig(
            name: 'cfg_json',
            source: {
              'type': 'embedded',
              'data': json.encode({'ok': true}),
            },
            typeHint: 'json',
            required: true,
          ),
          AssetConfig(
            name: 'lut_bin',
            source: {'type': 'embedded', 'bytes': 'AQIDBA=='},
            required: true,
          ),
        ],
        interPipelineLinks: const [],
        schemaVersion: 1,
      );

      final runtime = builder.build(net);
      expect(runtime.pipelines.length, 1);

      final pipeline = runtime.pipelines.first;

      // Start should preload network-level assets into the same shared manager
      await pipeline.start();

      final cfg = shared.getPreloadedConfig('cfg_json')!;
      final lut = shared.getPreloadedBinary('lut_bin')!;
      expect(cfg['ok'], isTrue);
      expect(lut, [1, 2, 3, 4]);

      await pipeline.stop();
      await shared.dispose();
    });
  });
}

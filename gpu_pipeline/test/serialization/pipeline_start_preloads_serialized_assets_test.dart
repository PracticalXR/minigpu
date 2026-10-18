import 'dart:convert';
import 'package:gpu_pipeline/src/serialization/pipeline_models.dart';
import 'package:test/test.dart';

import 'package:gpu_pipeline/src/pipeline.dart';
import 'package:gpu_pipeline/src/pipeline_assets.dart';

void main() {
  group('Pipeline + serialized assets integration', () {
    test(
      'Pipeline.start preloads AssetConfig set via setSerializedAssets',
      () async {
        final assets = AssetManager();

        // Create a pipeline and inject the shared AssetManager
        final pipe = Pipeline(id: 'p1', assetManager: assets);

        // Provide serialized assets (the "serialization system" side)
        final serialized = <AssetConfig>[
          AssetConfig(
            name: 'cfg_json',
            source: {
              'type': 'embedded',
              'data': json.encode({'a': 1}),
            },
            typeHint: 'json',
            required: true,
          ),
          AssetConfig(
            name: 'lut_bin',
            source: {'type': 'embedded', 'data': 'AQIDBA=='}, // base64
            required: true,
          ),
        ];
        // This assumes Pipeline exposes setSerializedAssets(List<AssetConfig>)
        pipe.setSerializedAssets(serialized);

        // Start pipeline: should preload serialized assets into the shared manager
        await pipe.start();

        // Validate assets are materialized and cached
        final cfg = assets.getPreloadedConfig('cfg_json')!;
        final lut = assets.getPreloadedBinary('lut_bin')!;
        expect(cfg['a'], 1);
        expect(lut.length, 4);

        // Cleanup
        await pipe.stop();
        await assets.dispose();
      },
    );
  });
}

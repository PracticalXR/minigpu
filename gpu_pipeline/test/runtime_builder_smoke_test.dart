import 'package:test/test.dart';
import 'package:gpu_pipeline/src/serialization/runtime_builder.dart';
import 'package:gpu_pipeline/src/serialization/pipeline_models.dart';
import 'package:gpu_pipeline/src/pipeline_assets.dart';

void main() {
  group('RuntimeBuilder', () {
    test('injects shared AssetManager and builds empty network', () {
      final shared = AssetManager();
      final builder = RuntimeBuilder(sharedAssets: shared);

      final net = NetworkConfig(
        // minimal empty network
        pipelines: const [],
        assets: const [],
        interPipelineLinks: const [],
        schemaVersion: 1,
      );

      final result = builder.build(net);

      expect(result.pipelines, isEmpty);
      expect(identical(result.assets, shared), isTrue);
    });
  });
}

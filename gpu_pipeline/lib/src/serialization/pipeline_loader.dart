import 'pipeline_serializer.dart';
import 'runtime_builder.dart';
import 'pipeline_models.dart';
import 'pipeline_validator.dart';
import '../pipeline_assets.dart';

class PipelineLoadResult {
  final NetworkConfig config;
  final ValidationReport validation;
  final RuntimeBuildResult runtime;
  final AssetManager assets;
  PipelineLoadResult({
    required this.config,
    required this.validation,
    required this.runtime,
    required this.assets,
  });

  Future<void> dispose() async {
    await runtime.dispose();
    await assets.dispose();
  }
}

class PipelineLoader {
  final PipelineSerializer _serializer = PipelineSerializer();
  final PipelineValidator _validator = PipelineValidator();

  PipelineLoadResult loadFromJson(String jsonStr) {
    final parsed = _serializer.parse(jsonStr);
    final config = parsed.config;
    final validation = parsed.report;

    final assets = AssetManager();
    final builder = RuntimeBuilder(sharedAssets: assets);
    final runtime = builder.build(config);

    return PipelineLoadResult(
      config: config,
      validation: validation,
      runtime: runtime,
      assets: assets,
    );
  }
}

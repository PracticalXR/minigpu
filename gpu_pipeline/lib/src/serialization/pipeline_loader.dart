import 'pipeline_serializer.dart';
import 'runtime_builder.dart';
import 'pipeline_models.dart';
import 'pipeline_validator.dart';
import '../pipeline_assets.dart';

/// Thrown by [PipelineLoader.loadFromJson] for a network with validation
/// errors. Nothing has been built.
class PipelineValidationException implements Exception {
  PipelineValidationException(this.report);

  /// The validation report, warnings included.
  final ValidationReport report;

  @override
  String toString() {
    final lines = [
      for (final e in report.errors) '  ${e.path}: ${e.code}: ${e.message}',
    ];
    return 'PipelineValidationException: the network has '
        '${report.errors.length} error(s)\n${lines.join('\n')}';
  }
}

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

  /// Disposes the pipelines and the links between them, then the asset
  /// manager.
  Future<void> dispose() async {
    await runtime.dispose();
    await assets.dispose();
  }
}

class PipelineLoader {
  final PipelineSerializer _serializer = PipelineSerializer();

  /// Parses and validates [jsonStr], then builds its pipelines.
  ///
  /// Throws a [PipelineValidationException] if validation finds an error.
  /// Warnings are in [PipelineLoadResult.validation].
  PipelineLoadResult loadFromJson(String jsonStr) {
    final parsed = _serializer.parse(jsonStr);
    final config = parsed.config;
    final validation = parsed.report;
    if (!validation.ok) throw PipelineValidationException(validation);

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

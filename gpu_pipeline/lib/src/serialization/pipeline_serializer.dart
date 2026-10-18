import 'dart:convert';
import 'pipeline_models.dart';
import 'pipeline_validator.dart';

class PipelineSerializationResult {
  final NetworkConfig config;
  final ValidationReport report;
  PipelineSerializationResult(this.config, this.report);
}

class PipelineSerializer {
  final PipelineValidator _validator = PipelineValidator();

  PipelineSerializationResult parse(String jsonStr) {
    final map = jsonDecode(jsonStr) as Map<String, dynamic>;
    final config = NetworkConfig.fromJson(map);
    final report = _validator.validate(config);
    return PipelineSerializationResult(config, report);
  }

  String encode(NetworkConfig config) => jsonEncode(config.toJson());

  // Convenience: parse -> re-encode (round-trip) for editor tests
  String normalize(String jsonStr) => encode(parse(jsonStr).config);
}

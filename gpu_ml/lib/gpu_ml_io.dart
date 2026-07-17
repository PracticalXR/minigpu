/// dart:io companion to gpu_ml: streaming GGUF access + the Qwen3.5/3.6
/// model runner for large local model files.  Web builds must not import
/// this library; use [GgufFile.parse] with in-memory bytes there.
library;

export 'gpu_ml.dart';
export 'src/gguf_stream.dart';
export 'src/qwen35_runner.dart';

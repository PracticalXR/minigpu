@TestOn('vm')
@Timeout(Duration(minutes: 5))
library;

import 'dart:io';

import 'package:test/test.dart';

/// Compiles example/web_smoke.dart with BOTH web compilers.  The entrypoint
/// pulls the entire web-safe gpu_ml surface (and transitively gpu_tensor +
/// minigpu's js_interop web bindings) through compilation, so any
/// web-breaking change (dart:io leaking into gpu_ml.dart, legacy js_util
/// interop, dart2wasm-incompatible constructs) fails HERE instead of in a
/// downstream app build.
void main() {
  final outDir = Directory.systemTemp.createTempSync('gpu_ml_web_smoke');
  tearDownAll(() {
    try {
      outDir.deleteSync(recursive: true);
    } catch (_) {}
  });

  Future<void> compile(List<String> args) async {
    final result = await Process.run(
      Platform.resolvedExecutable,
      args,
      workingDirectory: Directory.current.path,
    );
    expect(result.exitCode, equals(0),
        reason: 'stdout:\n${result.stdout}\nstderr:\n${result.stderr}');
  }

  test('dart2js compiles the web surface', () async {
    await compile([
      'compile',
      'js',
      'example/web_smoke.dart',
      '-o',
      '${outDir.path}${Platform.pathSeparator}smoke.js',
    ]);
  });

  test('dart2wasm compiles the web surface', () async {
    await compile([
      'compile',
      'wasm',
      'example/web_smoke.dart',
      '-o',
      '${outDir.path}${Platform.pathSeparator}smoke.wasm',
    ]);
  });
}

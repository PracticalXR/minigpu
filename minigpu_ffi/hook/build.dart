import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';
import 'package:logging/logging.dart';
import 'package:native_toolchain_cmake/native_toolchain_cmake.dart';

// Needs web conditional import
import 'dart:io';

final sourceDir = Directory('./src');

void main(List<String> args) async {
  await build(args, (input, output) async {
    Logger logger = Logger('build');
    await runBuild(input, output, sourceDir.absolute.uri);
    final minigpuLib = await output.findAndAddCodeAssets(input, names: {
      'minigpu_ffi': 'minigpu_ffi_bindings.dart',
    });
    final webgpuLib = await output.findAndAddCodeAssets(input,
        names: {
          'webgpu_dawn': 'webgpu_dawn.dart',
        },
        outDir: sourceDir.absolute.uri.resolve('external'));
    final assets = <List<dynamic>>[minigpuLib, webgpuLib];

    for (final assetList in assets) {
      for (CodeAsset asset in assetList) {
        logger.info('Added file: ${asset.file}');
      }
    }
  });
}

const name = 'mingpu_ffi.dart';

Future<void> runBuild(
  BuildInput input,
  BuildOutputBuilder output,
  Uri sourceDir,
) async {
  Generator generator = Generator.defaultGenerator;
  switch (input.config.code.targetOS) {
    case OS.android:
      generator = Generator.ninja;
      break;
    case OS.iOS:
      generator = Generator.make;
      break;
    case OS.macOS:
      generator = Generator.make;
      break;
    case OS.linux:
      generator = Generator.ninja;
      break;
    case OS.windows:
      generator = Generator.defaultGenerator;
      break;
    case OS.fuchsia:
      generator = Generator.defaultGenerator;
      break;
  }

  final builder = CMakeBuilder.create(
    name: name,
    sourceDir: sourceDir,
    generator: generator,
    defines: {},
  );
  await builder.run(
    input: input,
    output: output,
    logger: Logger('')
      ..level = Level.ALL
      ..onRecord.listen((record) => stderr.writeln(record)),
  );
}

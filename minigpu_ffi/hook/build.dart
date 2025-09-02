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

    // Removed dangling expression and restrict Dawn search to arch/OS-specific dir.
    final minigpuLib = await output.findAndAddCodeAssets(
      input,
      names: {'minigpu_ffi': 'minigpu_ffi_bindings.dart'},
    );

    final dawnNativeOutDir = _dawnNativeOutDir(input, sourceDir.absolute.uri);
    final webgpuLib = await output.findAndAddCodeAssets(
      input,
      names: {'webgpu_dawn': 'webgpu_dawn.dart'},
      outDir: dawnNativeOutDir,
    );

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
    buildMode: BuildMode.release,
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

Uri _dawnNativeOutDir(BuildInput input, Uri srcDir) {
  final osKey = _osKey(input.config.code.targetOS);
  final archKey = _archKey(input.config.code.targetArchitecture, input.config.code.targetOS);
  // Matches: ${DAWN_DIR}/build_${os}_${arch}/src/dawn/native
  return srcDir
      .resolve('external/')
      .resolve('dawn/')
      .resolve('build_${osKey}_${archKey}/')
      .resolve('src/dawn/native/');
}

String _osKey(OS os) => switch (os) {
      OS.android => 'android',
      OS.iOS => 'ios',
      OS.macOS => 'mac',
      OS.linux => 'unix',
      OS.windows => 'win',
      OS.fuchsia => 'unix', // fallback
      _ => 'unknown',
    };

String _archKey(Architecture arch, OS os) {
  // Mirror dawn.cmake normalization
  if (os == OS.android) {
    if (arch == Architecture.arm64) return 'arm64-v8a';
    if (arch == Architecture.arm) return 'armeabi-v7a';
    if (arch == Architecture.x64) return 'x86_64';
    if (arch == Architecture.ia32) return 'x86';
  } else {
    if (arch == Architecture.x64) return 'x86_64';
    if (arch == Architecture.arm64) return 'arm64';
    if (arch == Architecture.arm) return 'armv7';
  }
  return arch.name; // fallback
}

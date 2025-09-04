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
    if (input.config.code.targetOS != OS.iOS) {
      // Only search/runtime-package Dawn on platforms where it’s a shared library
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
    } else {
      // iOS: linked statically, nothing to add as an asset
      for (final asset in minigpuLib) {
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
      // Use Xcode so deployment target sticks
      generator = Generator.ninja;
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

  // Map Dart arch to CMake arch for iOS
  String? cmakeArch;
  if (input.config.code.targetOS == OS.iOS) {
    final a = input.config.code.targetArchitecture;
    if (a == Architecture.arm64) cmakeArch = 'arm64';
    if (a == Architecture.x64) cmakeArch = 'x86_64';
  }

  final builder = CMakeBuilder.create(
    name: name,
    sourceDir: sourceDir,
    generator: generator,
    buildMode: BuildMode.release,
    targets: [
      'minigpu_ffi',
      input.config.code.targetOS == OS.iOS ? 'webgpu_dawn' : null,
    ].whereType<String>().toList(),
    defines: {
      if (input.config.code.targetOS == OS.iOS && cmakeArch != null)
        'ENABLE_ARC': 'OFF',
    },
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
  final osKey = _osKey(input);
  final archKey = _archKey(
    input.config.code.targetArchitecture,
    input.config.code.targetOS,
  );
  return srcDir
      .resolve('external/')
      .resolve('dawn/')
      .resolve('build_${osKey}_${archKey}/');
}

String _osKey(BuildInput input) {
  final os = input.config.code.targetOS;
  if (os == OS.iOS) {
    // Prefer SDK info if available to distinguish simulator vs device
    try {
      // code_assets exposes iOS SDK in the config package
      final sdk =
          input.config.code.iOS.targetSdk; // IosSdk.device | IosSdk.simulator
      if (sdk.toString().contains('simulator')) return 'iossim';
      return 'ios';
    } catch (_) {
      // Heuristic fallback: x86_64 is always simulator; arm64 could be either
      final arch = input.config.code.targetArchitecture;
      if (arch.name == 'ia32' || arch.name == 'x86_64') return 'iossim';
      // Default to device when unknown
      return 'ios';
    }
  }
  return switch (os) {
    OS.android => 'android',
    OS.macOS => 'mac',
    OS.linux => 'unix',
    OS.windows => 'win',
    OS.fuchsia => 'unix',
    _ => 'unix',
  };
}

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

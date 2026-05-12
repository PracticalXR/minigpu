import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';
import 'package:logging/logging.dart';
import 'package:native_toolchain_cmake/native_toolchain_cmake.dart';
import 'package:path/path.dart' as p;

import 'dart:io';

final sourceDir = Directory('./src');

void main(List<String> args) async {
  await build(args, (input, output) async {
    if (!input.config.buildCodeAssets) return;

    final logger = Logger('build');
    _clearStaleCmakeCache(
      input.outputDirectory,
      sourceDir.absolute.uri,
      logger,
    );
    await runBuild(input, output, sourceDir.absolute.uri);

    final minigpuLib = await output.findAndAddCodeAssets(
      input,
      names: {'minigpu_ffi': 'minigpu_ffi_bindings.dart'},
    );

    final dawnSearchDirs = _dawnSearchDirs(input, sourceDir.absolute.uri);
    if (input.config.code.targetOS != OS.iOS) {
      // Find the pre-built Dawn shared library. We search multiple directories
      // in priority order:
      //   1. MINIGPU_DAWN_DIR env var (explicit override)
      //   2. Platform AppData / data-home (shared across all projects)
      //   3. Package-nested src/external/dawn/ (local dev fallback)
      //
      // Once found, copy it into input.outputDirectory alongside the built
      // minigpu_ffi library, then register it as a code asset.
      //
      // Registering a CodeAsset whose file lives outside outputDirectory is not
      // reliable -- some versions of the hooks runner only bundle files that
      // originate from outputDirectory. Copying first guarantees the DLL ends
      // up in the final app output.
      final dawnDll = _findDawnDll(input, dawnSearchDirs, logger);
      if (dawnDll == null) {
        final searched = dawnSearchDirs
            .map((u) => '  - ${Directory.fromUri(u).path}')
            .join('\n');
        final systemDir = _systemDawnRoot();
        final osKey = _osKey(input);
        final archKey = _archKey(
          input.config.code.targetArchitecture,
          input.config.code.targetOS,
        );
        throw StateError(
          'webgpu_dawn shared library not found.\n'
          'Searched:\n$searched\n\n'
          'To fix, pre-build Dawn for '
          '${input.config.code.targetOS}/${input.config.code.targetArchitecture}'
          ' and place the output in one of:\n'
          '  A) Set MINIGPU_DAWN_DIR=<dawn-root> (env var), then put the\n'
          '     build_${osKey}_$archKey/ folder inside it.\n'
          '  B) Copy build_${osKey}_$archKey/ to:\n'
          '     ${systemDir != null ? p.join(systemDir, 'build_${osKey}_$archKey') : "(system dir unavailable)"}\n'
          '  C) Place it at: '
          '${sourceDir.absolute.uri.resolve('external/dawn/build_${osKey}_$archKey/').toFilePath()}',
        );
      }

      // Copy into outputDirectory so the hooks runner bundles it reliably.
      final destFile = File.fromUri(
        input.outputDirectory.resolve(p.basename(dawnDll.path)),
      );
      if (!destFile.existsSync() ||
          destFile.lastModifiedSync() != dawnDll.lastModifiedSync()) {
        dawnDll.copySync(destFile.path);
        logger.info('Copied ${dawnDll.path} -> ${destFile.path}');
      }

      // Declare the source DLL as a build dependency so the hook re-runs
      // whenever Dawn is rebuilt.
      output.dependencies.add(dawnDll.uri);

      // Register both libraries as code assets from outputDirectory.
      final webgpuLib = await output.findAndAddCodeAssets(
        input,
        names: {'webgpu_dawn': 'webgpu_dawn.dart'},
      );

      final assets = <List<dynamic>>[minigpuLib, webgpuLib];
      for (final assetList in assets) {
        for (CodeAsset asset in assetList) {
          logger.info('Added file: ${asset.file}');
        }
      }
    } else {
      // iOS: linked statically, nothing to add as an asset.
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
      if (input.config.code.targetOS == OS.iOS && cmakeArch != null)
        'CMAKE_OSX_ARCHITECTURES': cmakeArch,
      // Always pass the system Dawn root so dawn.cmake never falls back to
      // the pub-cache-nested path.  cmake receives this as -DDAWN_DIR=…
      // before any in-file set() calls, guaranteeing the correct directory
      // even when the cmake subprocess inherits a minimal environment.
      if (_systemDawnRoot() case final dawnRoot?) 'DAWN_DIR': dawnRoot,
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

/// Returns a prioritised list of directories to search for the pre-built Dawn
/// shared library, for the current target OS + architecture:
///
///   1. `MINIGPU_DAWN_DIR` env var — set this to a "dawn root" that contains a
///      `build_{os}_{arch}/` subdirectory.  Good for CI or custom builds.
///   2. Platform AppData / data-home (`%LOCALAPPDATA%\minigpu\dawn` on Windows,
///      `~/Library/Application Support/minigpu/dawn` on macOS,
///      `~/.local/share/minigpu/dawn` on Linux).  One Dawn build shared across
///      all projects on the machine.
///   3. Package-nested `src/external/dawn/` — the original location, kept as a
///      fallback for local development or when packaging Dawn with the source.
List<Uri> _dawnSearchDirs(BuildInput input, Uri srcDir) {
  final osKey = _osKey(input);
  final archKey = _archKey(
    input.config.code.targetArchitecture,
    input.config.code.targetOS,
  );
  final buildSubdir = 'build_${osKey}_$archKey/';
  final dirs = <Uri>[];

  // 1. Explicit env-var override.
  final envOverride = Platform.environment['MINIGPU_DAWN_DIR'];
  if (envOverride != null && envOverride.isNotEmpty) {
    dirs.add(Uri.directory(p.join(envOverride, buildSubdir)));
  }

  // 2. System-level shared location.
  final systemRoot = _systemDawnRoot();
  if (systemRoot != null) {
    dirs.add(Uri.directory(p.join(systemRoot, buildSubdir)));
  }

  // 3. Package-nested fallback (local dev / bundled source).
  dirs.add(srcDir.resolve('external/dawn/$buildSubdir'));

  return dirs;
}

/// Returns the platform-specific root directory for the shared Dawn builds,
/// or null if the platform provides no suitable location (e.g. Android, iOS).
///
/// The layout under this root mirrors the package-nested layout:
///   <root>/build_{os}_{arch}/   — same convention as src/external/dawn/
String? _systemDawnRoot() {
  if (Platform.isWindows) {
    final localAppData = Platform.environment['LOCALAPPDATA'];
    if (localAppData != null && localAppData.isNotEmpty) {
      return p.join(localAppData, 'minigpu', 'dawn');
    }
  } else if (Platform.isMacOS) {
    final home = Platform.environment['HOME'];
    if (home != null && home.isNotEmpty) {
      return p.join(home, 'Library', 'Application Support', 'minigpu', 'dawn');
    }
  } else if (Platform.isLinux) {
    final xdgData = Platform.environment['XDG_DATA_HOME'];
    if (xdgData != null && xdgData.isNotEmpty) {
      return p.join(xdgData, 'minigpu', 'dawn');
    }
    final home = Platform.environment['HOME'];
    if (home != null && home.isNotEmpty) {
      return p.join(home, '.local', 'share', 'minigpu', 'dawn');
    }
  }
  return null;
}

/// Searches [searchDirs] in order, recursively, for `webgpu_dawn.dll` /
/// `libwebgpu_dawn.so` / `libwebgpu_dawn.dylib` matching the current target OS.
/// Returns the first match found, or null.
File? _findDawnDll(BuildInput input, List<Uri> searchDirs, Logger logger) {
  final os = input.config.code.targetOS;
  final filename = switch (os) {
    OS.windows => 'webgpu_dawn.dll',
    OS.macOS => 'libwebgpu_dawn.dylib',
    OS.linux || OS.android => 'libwebgpu_dawn.so',
    _ => 'libwebgpu_dawn.so',
  };

  for (final dirUri in searchDirs) {
    final dir = Directory.fromUri(dirUri);
    if (!dir.existsSync()) {
      logger.fine('Dawn search dir does not exist, skipping: ${dir.path}');
      continue;
    }
    for (final entity in dir.listSync(recursive: true, followLinks: false)) {
      if (entity is File && p.basename(entity.path) == filename) {
        logger.info('Found Dawn at: ${entity.path}');
        return entity;
      }
    }
    logger.fine('$filename not found under ${dir.path}');
  }
  return null;
}

String _osKey(BuildInput input) {
  final os = input.config.code.targetOS;
  if (os == OS.iOS) {
    try {
      final sdk = input.config.code.iOS.targetSdk;
      if (sdk.toString().contains('simulator')) return 'iossim';
      return 'ios';
    } catch (_) {
      final arch = input.config.code.targetArchitecture;
      if (arch.name == 'ia32' || arch.name == 'x86_64') return 'iossim';
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
  if (os == OS.android) {
    if (arch == Architecture.arm64) return 'arm64-v8a';
    if (arch == Architecture.arm) return 'armv7';
    if (arch == Architecture.x64) return 'x86_64';
    if (arch == Architecture.ia32) return 'x86';
  } else {
    if (arch == Architecture.x64) return 'x86_64';
    if (arch == Architecture.arm64) return 'arm64';
    if (arch == Architecture.arm) return 'armv7';
  }
  return arch.name;
}

/// Deletes the CMake output directory when its cached source path no longer
/// matches [currentSourceDir].
///
/// CMake records the absolute source path in CMakeCache.txt. When the package
/// switches between a pub-cache copy and a local path override, that path
/// changes and CMake refuses to proceed:
///
///   "The source X does not match the source Y used to generate cache."
///
/// `flutter clean` only removes `build/`; it leaves `.dart_tool/hooks_runner/`
/// intact, so the stale cache survives. This helper detects the mismatch and
/// deletes the output directory before CMake runs, forcing a clean configure.
void _clearStaleCmakeCache(Uri outputDir, Uri currentSourceDir, Logger logger) {
  final cacheFile = File.fromUri(outputDir.resolve('CMakeCache.txt'));
  if (!cacheFile.existsSync()) return;

  final lines = cacheFile.readAsLinesSync();
  String? cachedSrc;
  for (final line in lines) {
    if (line.startsWith('CMAKE_HOME_DIRECTORY:INTERNAL=')) {
      final eq = line.indexOf('=');
      cachedSrc = eq >= 0 ? line.substring(eq + 1).trim() : null;
      break;
    }
  }
  if (cachedSrc == null) return;

  final cached = cachedSrc.replaceAll(r'\', '/').toLowerCase();
  final current = currentSourceDir
      .toFilePath()
      .replaceAll(r'\', '/')
      .toLowerCase();

  if (cached == current) return;

  logger.warning(
    '[minigpu_ffi] Stale CMakeCache.txt detected.\n'
    '  Cached source : $cachedSrc\n'
    '  Current source: ${currentSourceDir.toFilePath()}\n'
    '  Deleting output directory to force a clean CMake configure.',
  );

  final outDirectory = Directory.fromUri(outputDir);
  if (outDirectory.existsSync()) {
    outDirectory.deleteSync(recursive: true);
  }
}

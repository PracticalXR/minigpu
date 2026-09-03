/// Persistent shader cache tests.
///
/// Compiling WGSL to a backend shader is the single most expensive thing a
/// minigpu process does at startup — on Windows/D3D11 it runs through FXC,
/// whose optimiser cost grows superlinearly with kernel size. The cache turns
/// that into a once-per-machine cost. These tests check the two things that
/// matter about it: that it actually HITS across process launches, and that
/// every way it can go wrong produces a miss rather than a wrong pipeline or a
/// failed device.
///
/// MOST OF THIS RUNS IN SUBPROCESSES, deliberately. The native context is
/// process-global and so is the cache's in-memory state, so "does a second
/// launch get hits?" is not answerable from inside one test process — an
/// in-process second init could be served by state that a real second launch
/// would not have. `test/tool/shader_cache_probe.dart` is one launch; the
/// tests below run it repeatedly against a temp directory and read its JSON.
///
/// Every test points the cache at its own temp directory. The developer's real
/// cache (%LOCALAPPDATA% / ~/Library/Caches / $XDG_CACHE_HOME) is never
/// touched.
@TestOn('vm')
@Timeout(Duration(minutes: 10))
library;

import 'dart:convert';
import 'dart:io';

import 'package:minigpu/minigpu.dart';
import 'package:test/test.dart';

/// Package root — `flutter test` runs with the package directory as cwd, but
/// an IDE or a repo-root invocation may not, so resolve it from this file.
late final Directory _pkgRoot = () {
  var dir = Directory.current;
  for (var i = 0; i < 6; i++) {
    if (File('${dir.path}/pubspec.yaml').existsSync() &&
        File('${dir.path}/test/tool/shader_cache_probe.dart').existsSync()) {
      return dir;
    }
    final parent = dir.parent;
    if (parent.path == dir.path) break;
    dir = parent;
  }
  return Directory.current;
}();

/// Path to a real `dart` executable.
///
/// NOT `Platform.resolvedExecutable`: under `flutter test` that is
/// `flutter_tester.exe`, and handing it `run <script>` starts a Flutter engine
/// that fails to build an isolate and then sits there — the failure looks like
/// a hang, not a bad path. The Dart SDK ships inside the Flutter cache, so
/// derive it from FLUTTER_ROOT or by walking up from the tester binary, and
/// only fall back to PATH.
late final String _dartExe = () {
  final exeName = Platform.isWindows ? 'dart.exe' : 'dart';

  bool works(String path) {
    if (!File(path).existsSync()) return false;
    try {
      return Process.runSync(path, ['--version']).exitCode == 0;
    } catch (_) {
      return false;
    }
  }

  // Already a dart binary (plain `dart test` run).
  final resolved = Platform.resolvedExecutable;
  if (resolved.endsWith(exeName) && works(resolved)) return resolved;

  final candidates = <String>[];
  final flutterRoot = Platform.environment['FLUTTER_ROOT'];
  if (flutterRoot != null && flutterRoot.isNotEmpty) {
    candidates.add('$flutterRoot/bin/cache/dart-sdk/bin/$exeName');
  }
  // .../bin/cache/artifacts/engine/<platform>/flutter_tester.exe
  //          ^-- walk up to `cache`, then into dart-sdk.
  var dir = File(resolved).parent;
  for (var i = 0; i < 6; i++) {
    candidates.add('${dir.path}/dart-sdk/bin/$exeName');
    final parent = dir.parent;
    if (parent.path == dir.path) break;
    dir = parent;
  }
  for (final c in candidates) {
    if (works(c)) return c;
  }

  // Last resort: PATH. Process.run does NOT do PATHEXT resolution on Windows,
  // so the entries have to be expanded by hand.
  for (final entry in (Platform.environment['PATH'] ?? '').split(
    Platform.isWindows ? ';' : ':',
  )) {
    if (entry.isEmpty) continue;
    final c = '$entry${Platform.pathSeparator}$exeName';
    if (works(c)) return c;
  }
  throw StateError(
    'No usable dart executable found; resolvedExecutable=$resolved',
  );
}();

/// One probe launch = one cold-or-warm process against [dir].
///
/// Returns the decoded counters. Throws with the full output when the probe
/// fails, because a probe that dies silently would otherwise look like a cache
/// miss and quietly pass the wrong test.
Future<Map<String, dynamic>> _probe(
  Directory dir, {
  int kernels = 3,
  String? extraKey,
  int? capBytes,
  bool disable = false,
  bool verify = false,
  Map<String, String>? env,
}) async {
  final result = await Process.run(_dartExe, [
    'run',
    'test/tool/shader_cache_probe.dart',
    '--dir=${dir.path}',
    '--kernels=$kernels',
    if (extraKey != null) '--extra-key=$extraKey',
    if (capBytes != null) '--cap=$capBytes',
    if (disable) '--disable',
    if (verify) '--verify',
  ], workingDirectory: _pkgRoot.path, environment: env);

  final stdoutText = result.stdout as String;
  // The build-hook runner can print on the same line, so find the object
  // rather than assuming the JSON starts at column 0.
  final start = stdoutText.indexOf('{"hits"');
  if (result.exitCode != 0 || start < 0) {
    fail(
      'shader_cache_probe failed (exit ${result.exitCode})\n'
      '--- stdout ---\n$stdoutText\n--- stderr ---\n${result.stderr}',
    );
  }
  final end = stdoutText.indexOf('}', start);
  return jsonDecode(stdoutText.substring(start, end + 1))
      as Map<String, dynamic>;
}

Directory _tempDir(String name) {
  final d = Directory(
    '${Directory.systemTemp.path}/mgpu_sc_test_${name}_${pid}_'
    '${DateTime.now().microsecondsSinceEpoch}',
  );
  addTearDown(() {
    if (d.existsSync()) {
      try {
        d.deleteSync(recursive: true);
      } catch (_) {
        // A blob another process still holds open is not worth failing a test.
      }
    }
  });
  return d;
}

void main() {
  group('persistent shader cache — hits across process launches', () {
    test('a second launch hits every entry the first one stored', () async {
      final dir = _tempDir('warm');

      final cold = await _probe(dir, verify: true);
      expect(
        cold['stores'],
        greaterThan(0),
        reason: 'a cold launch must write compiled blobs to $dir',
      );
      expect(cold['hits'], 0, reason: 'nothing can hit on an empty cache');
      expect(cold['storeFailures'], 0);
      expect(cold['entryCount'], greaterThan(0));

      final warm = await _probe(dir, verify: true);
      expect(
        warm['hits'],
        cold['misses'],
        reason: 'every lookup the cold launch missed must hit on the second',
      );
      expect(
        warm['misses'],
        0,
        reason: 'a warm launch must not compile anything',
      );
      expect(
        warm['stores'],
        0,
        reason: 'nothing new to store when everything hit',
      );

      // Byte-identical output cold vs warm is the acceptance test that matters:
      // a shader cache that changes a single result value is a bug, however
      // fast it is.
      expect(
        warm['checksum'],
        cold['checksum'],
        reason: 'cached pipelines must produce byte-identical results',
      );

      // Reported, not asserted — absolute milliseconds on a loaded machine are
      // noisy enough that a threshold here would be a flake generator. The
      // counters above are the real assertion.
      // ignore: avoid_print
      print(
        'pipelineCreateMs cold=${cold['pipelineCreateMs']} '
        'warm=${warm['pipelineCreateMs']}',
      );
    });

    test('a changed key input misses instead of serving a stale blob', () async {
      final dir = _tempDir('stalekey');

      final base = await _probe(dir);
      expect(base['stores'], greaterThan(0));

      // extraKey stands in for any key input changing under the cache — a Dawn
      // version bump, a driver update, a different adapter. All of them must
      // land as a miss, never as a stale hit, because a stale blob is not a
      // slow frame: it is the wrong pipeline.
      final different = await _probe(dir, extraKey: 'pretend-new-driver');
      expect(
        different['hits'],
        0,
        reason: 'a different key must not reach entries written under another',
      );
      expect(different['misses'], base['misses']);
      expect(
        different['entryCount'],
        greaterThan(base['entryCount'] as int),
        reason: 'the new key writes its own entries alongside the old ones',
      );

      // ...and the original key still resolves to its own entries.
      final backToBase = await _probe(dir);
      expect(
        backToBase['hits'],
        base['misses'],
        reason: 'the first key must still hit after a second key wrote entries',
      );
      expect(backToBase['misses'], 0);
    });
  });

  group('persistent shader cache — environment overrides', () {
    // These exist so "is the cache the problem?" is answerable on a build you
    // cannot edit. That only holds if they beat whatever the app hard-coded,
    // so both tests below pass a directory programmatically and check the
    // environment wins anyway.

    test('MGPU_SHADER_CACHE=0 disables it whatever the app asked for', () async {
      final dir = _tempDir('env_off');

      final warmed = await _probe(dir, verify: true);
      expect(warmed['stores'], greaterThan(0));

      final off = await _probe(
        dir,
        verify: true,
        env: {'MGPU_SHADER_CACHE': '0'},
      );
      expect(off['enabled'], isFalse);
      expect(
        off['hits'],
        0,
        reason: 'a disabled cache must not read the entries already there',
      );
      expect(off['stores'], 0);
      expect(
        off['checksum'],
        warmed['checksum'],
        reason: 'disabling the cache must not change what the GPU computes',
      );

      // The env var must not have deleted or disturbed what was stored.
      final backOn = await _probe(dir, verify: true);
      expect(
        backOn['hits'],
        greaterThan(0),
        reason: 'entries must survive a run that had caching switched off',
      );
    });

    test('MGPU_SHADER_CACHE_DIR redirects storage', () async {
      final appDir = _tempDir('env_appdir');
      final envDir = _tempDir('env_envdir');

      final result = await _probe(
        appDir,
        verify: true,
        env: {'MGPU_SHADER_CACHE_DIR': envDir.path},
      );

      expect(
        result['directory'],
        envDir.path,
        reason: 'the environment must outrank the programmatic directory',
      );
      expect(result['stores'], greaterThan(0));
      expect(
        envDir.existsSync() ? envDir.listSync().length : 0,
        greaterThan(0),
        reason: 'blobs must land in the directory the environment named',
      );
      expect(
        appDir.existsSync() ? appDir.listSync().length : 0,
        0,
        reason: 'nothing may be written to the directory the app asked for',
      );
    });
  });

  group('persistent shader cache — concurrent processes', () {
    test('two cold launches racing one empty directory stay consistent',
        () async {
      final dir = _tempDir('race');

      // Both processes miss everything and then write the same keys at
      // roughly the same moment. Writes go to a per-process temp file and are
      // renamed into place, so the only possible outcomes are "both wrote" and
      // "one lost the rename race" — never a reader seeing a half-written
      // blob, and never a failure that propagates to the device.
      final results = await Future.wait([
        _probe(dir, verify: true),
        _probe(dir, verify: true),
      ]);

      for (final r in results) {
        expect(
          r['misses'],
          greaterThan(0),
          reason: 'both launches started from an empty cache',
        );
        expect(
          r['checksum'],
          results.first['checksum'],
          reason: 'a write race must not change what either process computes',
        );
      }

      // No temp file may survive a clean run: a leftover .tmp- would mean a
      // write path that neither renamed nor cleaned up.
      final leftovers = dir
          .listSync()
          .whereType<File>()
          .where((f) => f.path.contains('.tmp-'))
          .toList();
      expect(
        leftovers,
        isEmpty,
        reason: 'atomic writes must leave no temp files behind',
      );

      // Whatever the race did, the directory that resulted must be usable.
      final after = await _probe(dir, verify: true);
      expect(
        after['misses'],
        0,
        reason: 'a launch after the race must find every entry intact',
      );
      expect(after['checksum'], results.first['checksum']);
    });
  });

  group('persistent shader cache — every failure is a miss, not a crash', () {
    test('corrupt and truncated entries recompile without crashing', () async {
      final dir = _tempDir('corrupt');

      final cold = await _probe(dir, verify: true);
      expect(cold['stores'], greaterThan(0));

      // Mangle every entry three different ways: truncation, a flipped bit in
      // the payload, and a destroyed header.
      final files = dir.listSync().whereType<File>().toList()
        ..sort((a, b) => a.path.compareTo(b.path));
      expect(files, isNotEmpty);
      for (var i = 0; i < files.length; i++) {
        final bytes = files[i].readAsBytesSync();
        switch (i % 3) {
          case 0: // truncate to half
            files[i].writeAsBytesSync(bytes.sublist(0, bytes.length ~/ 2));
          case 1: // flip a payload bit
            bytes[bytes.length - 1] ^= 0xFF;
            files[i].writeAsBytesSync(bytes);
          case 2: // destroy the magic
            bytes[0] ^= 0xFF;
            files[i].writeAsBytesSync(bytes);
        }
      }

      final afterCorruption = await _probe(dir, verify: true);
      expect(
        afterCorruption['misses'],
        greaterThan(0),
        reason: 'a corrupt entry must be a miss',
      );
      expect(
        afterCorruption['checksum'],
        cold['checksum'],
        reason: 'recompiled pipelines must still produce the right answer',
      );
    });

    test('an unusable cache directory still yields a working device', () async {
      final dir = _tempDir('unusable');
      // Make the cache path's PARENT a regular file, so creating the directory
      // cannot succeed however the platform spells the error.
      dir.parent.createSync(recursive: true);
      final blocker = File(dir.path);
      blocker.writeAsStringSync('not a directory');
      addTearDown(() {
        if (blocker.existsSync()) blocker.deleteSync();
      });

      final result = await _probe(Directory('${dir.path}/sub'), verify: true);
      expect(
        result['stores'],
        0,
        reason: 'nothing can be stored under an unusable path',
      );
      // The device came up and computed the right answer anyway — which is the
      // whole contract. A cache that can break startup is worse than no cache.
      expect(result['checksum'], isNot(0));
    });

    test('disabled means no directory use and no stores', () async {
      final dir = _tempDir('disabled');

      final result = await _probe(dir, disable: true, verify: true);
      expect(result['enabled'], isFalse);
      expect(result['stores'], 0);
      expect(result['hits'], 0);
      expect(
        result['checksum'],
        isNot(0),
        reason: 'disabling the cache must not change what the GPU computes',
      );
      expect(
        dir.existsSync() ? dir.listSync().length : 0,
        0,
        reason: 'a disabled cache must not write anything',
      );
    });

    test('a cap smaller than one entry evicts without breaking anything',
        () async {
      final dir = _tempDir('evict');

      // 1 KiB is below the size of a single compiled blob, so every store
      // immediately pushes the total over and triggers eviction. Nothing may
      // crash, and the results must still be correct.
      final result = await _probe(dir, capBytes: 1024, verify: true);
      expect(result['stores'], greaterThan(0));
      expect(
        result['evictions'],
        greaterThan(0),
        reason: 'a cap this small must force eviction',
      );
      expect(result['checksum'], isNot(0));

      final reference = await _probe(_tempDir('evict_ref'), verify: true);
      expect(
        result['checksum'],
        reference['checksum'],
        reason: 'eviction must not affect what the GPU computes',
      );
    });
  });

  group('persistent shader cache — Dart API', () {
    test('configure, inspect and clear round-trip', () async {
      final dir = _tempDir('api');
      dir.createSync(recursive: true);

      // The context is process-global and another suite in this run may
      // already own one, in which case configuration is correctly reported as
      // too-late. Both outcomes are valid; what must hold is that the getters
      // agree with what was set.
      final preInit = Minigpu.configureShaderCache(
        directory: dir.path,
        maxBytes: 64 * 1024 * 1024,
      );
      expect(preInit, isA<bool>());

      expect(
        Minigpu.shaderCacheDirectory,
        dir.path,
        reason: 'the configured directory must be what the cache reports',
      );

      final stats = Minigpu.shaderCacheStats;
      expect(stats, isNotNull);
      expect(stats!.enabled, isTrue);
      expect(
        stats.usingDefaultProvider,
        isTrue,
        reason: 'no custom provider is installed by these tests',
      );
      expect(stats.toString(), contains('hits'));

      // Clearing an empty directory removes nothing and must not throw.
      expect(Minigpu.clearShaderCache(), 0);

      // Leave the process pointed back at its default so a later suite in the
      // same `dart test` run is unaffected.
      addTearDown(() {
        Minigpu.configureShaderCache(directory: '', maxBytes: 256 * 1024 * 1024);
      });
    });

    test('clear removes stored entries', () async {
      final dir = _tempDir('clear');
      final cold = await _probe(dir);
      expect(cold['stores'], greaterThan(0));
      expect(dir.listSync().whereType<File>(), isNotEmpty);

      Minigpu.configureShaderCache(directory: dir.path);
      final removed = Minigpu.clearShaderCache();
      expect(removed, greaterThan(0));
      expect(dir.listSync().whereType<File>(), isEmpty);

      addTearDown(() {
        Minigpu.configureShaderCache(directory: '');
      });
    });
  });
}

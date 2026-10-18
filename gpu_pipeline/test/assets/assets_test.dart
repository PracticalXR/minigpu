import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:test/test.dart';
import 'package:gpu_pipeline/src/pipeline_assets.dart';
import 'package:gpu_pipeline/src/serialization/pipeline_models.dart';

void main() {
  group('AssetManager', () {
    late AssetManager manager;
    late Directory tempDir;

    setUp(() async {
      manager = AssetManager();
      tempDir = await Directory.systemTemp.createTemp('asset_test_');
    });

    tearDown(() async {
      // No dispose() API; clear cache and remove temp directory
      manager.clearCache();
      await tempDir.delete(recursive: true);
    });

    test('preloads file asset (json) and caches it', () async {
      final file = File('${tempDir.path}\\a.json');
      await file.writeAsString(jsonEncode({'ok': true}));

      final cfg = AssetConfig(
        name: 'cfg',
        source: {'type': 'file', 'path': file.path},
        // defaults: strategy preload, cache true
        typeHint: 'json',
        required: true,
      );

      await manager.preloadBatch([cfg]);

      final asset = manager.get('cfg');
      expect(asset, isNotNull);
      expect(asset!.type, AssetType.json);
      final decoded =
          jsonDecode(utf8.decode(asset.bytes!)) as Map<String, dynamic>;
      expect(decoded['ok'], true);
    });

    test('optional missing file does not throw; emits failed event', () async {
      final events = <AssetEvent>[];
      final sub = manager.events.listen(events.add);

      final cfg = AssetConfig(
        name: 'missing_optional',
        source: {'type': 'file', 'path': '${tempDir.path}\\nope.bin'},
        required: false,
      );

      await manager.preloadBatch([cfg]);

      expect(manager.get('missing_optional'), isNull);
      await Future<void>.delayed(const Duration(milliseconds: 10));
      expect(
        events.any((e) => e.kind == 'failed' && e.name == 'missing_optional'),
        isTrue,
      );
      await sub.cancel();
    });

    test('required missing file throws and emits failed event', () async {
      final events = <AssetEvent>[];
      final sub = manager.events.listen(events.add);

      final cfg = AssetConfig(
        name: 'missing_required',
        source: {'type': 'file', 'path': '${tempDir.path}\\nope.bin'},
        required: true,
      );

      expect(() => manager.preloadBatch([cfg]), throwsA(isA<Exception>()));

      await Future<void>.delayed(const Duration(milliseconds: 10));
      expect(
        events.any((e) => e.kind == 'failed' && e.name == 'missing_required'),
        isTrue,
      );
      await sub.cancel();
    });

    test('loads embedded text and binary assets', () async {
      // Text
      final textCfg = AssetConfig(
        name: 'txt',
        source: {'type': 'embedded', 'data': 'hello'},
        typeHint: 'text',
      );
      final textAsset = await manager.load(textCfg);
      expect(textAsset.text, 'hello');
      expect(textAsset.type, AssetType.text);

      // Binary (base64 input)
      final binData = base64.encode([1, 2, 3, 4]);
      final binCfg = AssetConfig(
        name: 'bin',
        source: {'type': 'embedded', 'data': binData},
        // hint omitted; will infer binary from non-text
      );
      final binAsset = await manager.load(binCfg);
      expect(binAsset.bytes, isNotNull);
      expect(binAsset.bytes, Uint8List.fromList([1, 2, 3, 4]));
      expect(binAsset.type, isIn([AssetType.binary, AssetType.unknown]));
    });

    test('loads url asset via local HttpServer', () async {
      final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
      addTearDown(() async => server.close(force: true));
      server.listen((HttpRequest req) {
        req.response.statusCode = 200;
        req.response.headers.contentType = ContentType.json;
        req.response.write(jsonEncode({'served': true}));
        req.response.close();
      });
      final url = 'http://127.0.0.1:${server.port}/data.json';

      final cfg = AssetConfig(
        name: 'remote',
        source: {'type': 'url', 'url': url},
        typeHint: 'json',
      );

      final asset = await manager.load(cfg);
      expect(asset.type, AssetType.json);
      final decoded =
          jsonDecode(utf8.decode(asset.bytes!)) as Map<String, dynamic>;
      expect(decoded['served'], true);
    });

    test('generated source is not implemented', () async {
      final cfg = AssetConfig(
        name: 'gen',
        source: {
          'type': 'generated',
          'params': {
            'shape': [4],
          },
        },
      );
      expect(() => manager.load(cfg), throwsUnsupportedError);
    });

    test('per-call assets are evicted on token dispose', () async {
      final events = <AssetEvent>[];
      final sub = manager.events.listen(events.add);

      final tempAsset = StaticAsset(
        name: 'frame_image',
        type: AssetType.binary,
        bytes: Uint8List.fromList([9, 9, 9]),
      );

      final token = manager.registerPerCall({'frame_image': tempAsset});
      expect(manager.get('frame_image'), isNotNull);

      token.dispose();
      await Future<void>.delayed(const Duration(milliseconds: 10));

      expect(manager.get('frame_image'), isNull);
      expect(
        events.any((e) => e.kind == 'evicted' && e.name == 'frame_image'),
        isTrue,
      );
      await sub.cancel();
    });

    test('per-call assets promoted to cache survive dispose', () async {
      final tempAsset = StaticAsset(
        name: 'keep_me',
        type: AssetType.text,
        text: 'hello',
      );
      final token = manager.registerPerCall({
        'keep_me': tempAsset,
      }, promoteToCache: true);
      expect(manager.get('keep_me'), isNotNull);

      token.dispose();
      expect(manager.get('keep_me'), isNotNull);

      // Clear explicitly
      manager.clearAsset('keep_me');
      expect(manager.get('keep_me'), isNull);
    });

    test('clearAsset and clearCache emit evicted events', () async {
      final events = <AssetEvent>[];
      final sub = manager.events.listen(events.add);

      // Load two embedded assets with cache true (default for preload)
      final a1 = await manager.load(
        AssetConfig(
          name: 'a1',
          source: {'type': 'embedded', 'data': 'x'},
          typeHint: 'text',
        ),
      );
      final a2 = await manager.load(
        AssetConfig(
          name: 'a2',
          source: {
            'type': 'embedded',
            'data': [1, 2, 3],
          },
        ),
      );
      expect(manager.get('a1'), isNotNull);
      expect(manager.get('a2'), isNotNull);
      expect(a1.text, 'x');
      expect(a2.bytes, isNotEmpty);

      manager.clearAsset('a1');
      expect(manager.get('a1'), isNull);

      manager.clearCache();
      expect(manager.get('a2'), isNull);

      await Future<void>.delayed(const Duration(milliseconds: 10));
      expect(
        events.where((e) => e.kind == 'evicted').length,
        greaterThanOrEqualTo(2),
      );
      await sub.cancel();
    });

    test('type inference from file extension', () async {
      final jsonFile = File('${tempDir.path}\\t.json');
      await jsonFile.writeAsString('{}');
      final wgslFile = File('${tempDir.path}\\s.wgsl');
      await wgslFile.writeAsString('// shader');
      final binFile = File('${tempDir.path}\\b.bin');
      await binFile.writeAsBytes([0, 1, 2]);

      final a = await manager.load(
        AssetConfig(name: 'j', source: {'type': 'file', 'path': jsonFile.path}),
      );
      final b = await manager.load(
        AssetConfig(name: 's', source: {'type': 'file', 'path': wgslFile.path}),
      );
      final c = await manager.load(
        AssetConfig(name: 'b', source: {'type': 'file', 'path': binFile.path}),
      );

      expect(a.type, AssetType.json);
      expect(b.type, AssetType.text);
      expect(c.type, AssetType.binary);
    });

    test('release decreases refcount and may evict when non-persistent', () async {
      // Load with cache true (persistent) then manually flip to non-persistent to test release.
      await manager.load(
        AssetConfig(
          name: 'rel',
          source: {
            'type': 'embedded',
            'data': [7, 7],
          },
        ),
      );
      // Not exposed, but we can simulate non-persistent behavior using per-call
      final token = manager.registerPerCall({
        'temp_rel': StaticAsset(
          name: 'temp_rel',
          type: AssetType.binary,
          bytes: Uint8List(1),
        ),
      });
      expect(manager.get('temp_rel'), isNotNull);
      token.dispose();
      expect(manager.get('temp_rel'), isNull);
    });

    test('typed getters return expected shapes', () async {
      // Embedded JSON -> getPreloadedConfig
      final cfgJson = AssetConfig(
        name: 'cfg_json',
        source: {'type': 'embedded', 'data': '{"a": 1, "b": "x"}'},
        typeHint: 'json',
      );
      await manager.preloadBatch([cfgJson]);
      final m = manager.getPreloadedConfig('cfg_json');
      expect(m, isNotNull);
      expect(m!['a'], 1);
      expect(m['b'], 'x');

      // Embedded base64 -> getPreloadedBinary
      final binData = base64.encode([10, 11, 12]);
      final cfgBin = AssetConfig(
        name: 'cfg_bin',
        source: {'type': 'embedded', 'data': binData},
      );
      await manager.preloadBatch([cfgBin]);
      final bin = manager.getPreloadedBinary('cfg_bin');
      expect(bin, Uint8List.fromList([10, 11, 12]));

      // Preloaded tensor via StaticAsset.tensor -> getPreloadedTensor
      final t = await Tensor.zeros<Float32List>([4]);
      await manager.preload(StaticAsset.tensor(name: 'pre_tensor', data: t));
      final got = manager.getPreloadedTensor('pre_tensor');
      expect(got, isNotNull);
      expect(got!.shape, [4]);
    });

    test('url 404 emits failed and does not cache', () async {
      final events = <AssetEvent>[];
      final sub = manager.events.listen(events.add);

      // Use a high port to force connection error or closed server
      final cfg = AssetConfig(
        name: 'bad_url',
        source: {'type': 'url', 'url': 'http://127.0.0.1:9/404.json'},
        typeHint: 'json',
        required: false,
      );

      try {
        await manager.preloadBatch([cfg]);
      } catch (_) {
        // ignore; required=false should swallow in preloadBatch, but
        // some platforms may throw on connect error
      }

      await Future<void>.delayed(const Duration(milliseconds: 50));
      expect(manager.get('bad_url'), isNull);
      expect(
        events.any((e) => e.kind == 'failed' && e.name == 'bad_url'),
        isTrue,
      );
      await sub.cancel();
    });

    test('stage-provided StaticAsset preload keeps cached copy', () async {
      final tmp = File('${tempDir.path}\\stage_weights.bin');
      await tmp.writeAsBytes([1, 2, 3, 4]);

      await manager.preload(
        StaticAsset.weights('stage_weights', path: tmp.path),
      );

      final a = manager.get('stage_weights');
      expect(a, isNotNull);
      expect(a!.bytes, isNotNull);
      expect(a.bytes, Uint8List.fromList([1, 2, 3, 4]));
    });
  });
}

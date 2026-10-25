import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

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
      await manager.dispose();
      await tempDir.delete(recursive: true);
    });

    test('preloads file asset (json) and caches it', () async {
      final file = File('${tempDir.path}\\cfg.json');
      await file.writeAsString(json.encode({'ok': true}));

      final cfg = AssetConfig(
        name: 'cfg',
        source: {'type': 'file', 'path': file.path},
        typeHint: 'json',
        required: true,
      );

      await manager.preloadBatch([cfg]);

      final decoded = manager.getPreloadedConfig('cfg');
      expect(decoded, isNotNull);
      expect(decoded!['ok'], true);
    });

    test('optional missing file does not throw; emits failed event', () async {
      final cfg = AssetConfig(
        name: 'missing_opt',
        source: {'type': 'file', 'path': '${tempDir.path}\\nope.json'},
        typeHint: 'json',
        required: false,
      );

      final events = <AssetEvent>[];
      final sub = manager.events.listen(events.add);

      await manager.preloadBatch([cfg]);

      expect(
        events.any((e) => e.kind == 'failed' && e.name == 'missing_opt'),
        isTrue,
      );
      expect(manager.get('missing_opt'), isNull);

      await sub.cancel();
    });

    test('required missing file throws and emits failed event', () async {
      final cfg = AssetConfig(
        name: 'missing_req',
        source: {'type': 'file', 'path': '${tempDir.path}\\nope.json'},
        typeHint: 'json',
        required: true,
      );

      final events = <AssetEvent>[];
      final sub = manager.events.listen(events.add);

      expect(() => manager.preloadBatch([cfg]), throwsA(anything));

      // Give event loop a tick to capture event if thrown asynchronously
      await Future<void>.delayed(const Duration(milliseconds: 10));

      expect(
        events.any((e) => e.kind == 'failed' && e.name == 'missing_req'),
        isTrue,
      );

      await sub.cancel();
    });

    test('loads embedded text and binary assets', () async {
      // Text embedded
      final textCfg = AssetConfig(
        name: 'hello',
        source: {'type': 'embedded', 'data': 'hello world'},
        typeHint: 'text',
      );
      await manager.load(textCfg);
      final cfgMap = manager.getPreloadedConfig('hello');
      expect(cfgMap, isNull); // not JSON
      final textAsset = manager.get('hello');
      expect(textAsset, isNotNull);
      expect(textAsset!.text, 'hello world');

      // Binary embedded via base64
      final binCfg = AssetConfig(
        name: 'lut',
        source: {
          'type': 'embedded',
          'data': base64.encode([1, 2, 3, 4]),
        },
        typeHint: 'binary',
      );
      await manager.load(binCfg);
      final bin = manager.getPreloadedBinary('lut');
      expect(bin, isNotNull);
      expect(bin, equals(Uint8List.fromList([1, 2, 3, 4])));
    });

    test('loads url asset via local HttpServer', () async {
      final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
      addTearDown(() => server.close(force: true));

      server.listen((HttpRequest req) async {
        req.response.headers.contentType = ContentType.json;
        req.response.write(json.encode({'served': true}));
        await req.response.close();
      });

      final cfg = AssetConfig(
        name: 'remote_cfg',
        source: {
          'type': 'url',
          'url': 'http://localhost:${server.port}/cfg.json',
        },
        typeHint: 'json',
      );

      await manager.load(cfg);

      final decoded = manager.getPreloadedConfig('remote_cfg');
      expect(decoded, isNotNull);
      expect(decoded!['served'], true);
    });

    test('generated source is not implemented', () async {
      final cfg = AssetConfig(
        name: 'gen',
        source: {
          'type': 'generated',
          'params': {
            'shape': [2, 2],
          },
        },
      );
      expect(() => manager.load(cfg), throwsUnsupportedError);
    });

    test('per-call assets are evicted on token dispose', () async {
      final events = <AssetEvent>[];
      final sub = manager.events.listen(events.add);

      final token = manager.registerPerCall({
        'frame': StaticAsset.bytes(
          name: 'frame',
          type: AssetType.binary,
          data: Uint8List.fromList([9, 8, 7]),
        ),
      });

      expect(manager.get('frame'), isNotNull);

      await token.dispose();

      expect(manager.get('frame'), isNull);
      // Should observe released/evicted events
      expect(
        events.any((e) => e.kind == 'evicted' && e.name == 'frame'),
        isTrue,
      );

      await sub.cancel();
    });
  });
}

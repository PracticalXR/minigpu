import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:gpu_pipeline/src/pipeline_assets.dart';

Uint8List _bytes(int n) =>
    Uint8List.fromList(List<int>.generate(n, (i) => i & 0xFF));

void main() {
  group('AssetManager cache limit (LRU, persistent, per-call)', () {
    late AssetManager m;
    late List<AssetEvent> events;

    setUp(() {
      m = AssetManager();
      events = [];
      m.events.listen(events.add);
    });

    tearDown(() async {
      await m.dispose();
    });

    test(
      'persistent assets survive cache limit (no eviction of persistent)',
      () async {
        await m.setCacheLimit(20); // 20 bytes

        await m.preload(
          StaticAsset.bytes(
            name: 'a',
            type: AssetType.binary,
            data: _bytes(16),
          ),
        );
        await m.preload(
          StaticAsset.bytes(
            name: 'b',
            type: AssetType.binary,
            data: _bytes(16),
          ),
        );

        // Both are persistent by design; limit should not evict them.
        expect(m.get('a'), isNotNull);
        expect(m.get('b'), isNotNull);

        // No evicted events were emitted
        expect(events.where((e) => e.kind == 'evicted'), isEmpty);
      },
    );

    test(
      'per-call non-persistent is not evicted by limit while in-use',
      () async {
        await m.setCacheLimit(16); // small limit

        final token = m.registerPerCall({
          'frame': StaticAsset.bytes(
            name: 'frame',
            type: AssetType.binary,
            data: _bytes(64), // exceeds limit
          ),
        });

        // Still present because refCount == 1 (in-use)
        expect(m.get('frame'), isNotNull);

        // Disposing per-call token releases and evicts it immediately
        await token.dispose();
        expect(m.get('frame'), isNull);

        final evicted = events.where(
          (e) => e.kind == 'evicted' && e.name == 'frame',
        );
        expect(evicted.isNotEmpty, isTrue);
      },
    );

    test('touch (get) maintains recency without errors', () async {
      await m.setCacheLimit(32);

      await m.preload(
        StaticAsset.bytes(name: 'p1', type: AssetType.binary, data: _bytes(16)),
      );
      await m.preload(
        StaticAsset.bytes(name: 'p2', type: AssetType.binary, data: _bytes(16)),
      );

      // Touch both; recency is internal, just assert presence and no evictions.
      expect(m.get('p1'), isNotNull);
      expect(m.get('p2'), isNotNull);
      expect(events.where((e) => e.kind == 'evicted'), isEmpty);
    });

    test('setCacheLimit(0) disables limit', () async {
      await m.setCacheLimit(0); // disabled

      await m.preload(
        StaticAsset.text(
          name: 't',
          data: 'x' * 1024, // > 0
        ),
      );

      expect(m.get('t'), isNotNull);
      expect(events.where((e) => e.kind == 'evicted'), isEmpty);
    });

    test('clearCache emits evicted for each cached asset', () async {
      await m.preload(
        StaticAsset.bytes(name: 'a1', type: AssetType.binary, data: _bytes(4)),
      );
      await m.preload(
        StaticAsset.bytes(name: 'a2', type: AssetType.binary, data: _bytes(4)),
      );

      // Loaded events for both
      final loaded = events.where((e) => e.kind == 'loaded').length;
      expect(loaded, greaterThanOrEqualTo(2));

      // Clear cache should emit evicted for both
      m.clearCache();
      final evicted = events.where((e) => e.kind == 'evicted').length;
      expect(evicted, greaterThanOrEqualTo(2));

      expect(m.get('a1'), isNull);
      expect(m.get('a2'), isNull);
    });
  });
}

import 'dart:io';
import 'dart:typed_data';

import 'package:gpu_pipeline/gpu_pipeline.dart';
import 'package:test/test.dart';

AssetConfig _text(String name, String text, {bool cache = true}) => AssetConfig(
  name: name,
  source: {'type': 'embedded', 'text': text},
  cache: cache,
);

StaticAsset _bytes(String name, int length) => StaticAsset.bytes(
  name: name,
  type: AssetType.binary,
  data: Uint8List(length),
);

const _missingFile = {'type': 'file', 'path': 'no/such/asset.bin'};

void main() {
  late AssetManager assets;
  late List<String> events;

  setUp(() {
    assets = AssetManager();
    events = [];
    assets.events.listen((e) => events.add('${e.kind} ${e.name}'));
  });

  tearDown(() => assets.dispose());

  group('AssetManager holders', () {
    test(
      'the cache limit evicts the least recently used unheld asset',
      () async {
        await assets.setCacheLimit(10);
        await assets.preload(_bytes('a', 4));
        await assets.preload(_bytes('b', 4));
        await assets.release('a');
        await assets.release('b');
        // Preloaded assets stay cached when nothing holds them.
        expect(assets.get('a'), isNotNull);
        expect(assets.get('b'), isNotNull);

        await assets.load(
          AssetConfig(
            name: 'c',
            source: {
              'type': 'embedded',
              'bytes': [1, 2, 3, 4],
            },
          ),
        );
        expect(assets.get('a'), isNull);
        expect(assets.get('b'), isNotNull);
        expect(assets.get('c'), isNotNull);
      },
    );

    test('a held asset is evicted only once it is released', () async {
      await assets.setCacheLimit(4);
      await assets.preload(_bytes('big', 8));
      expect(assets.get('big'), isNotNull);

      await assets.release('big');
      expect(assets.get('big'), isNull);
      expect(events, ['loaded big', 'released big', 'evicted big']);
    });

    test('an uncached asset leaves with its last holder', () async {
      final config = _text('x', 'hi', cache: false);
      await assets.load(config);
      await assets.load(config);

      await assets.release('x');
      expect(assets.get('x'), isNotNull);
      await assets.release('x');
      expect(assets.get('x'), isNull);
    });

    test('registerPerCall refuses a name that is in use', () async {
      await assets.preload(StaticAsset.text(name: 'lut', data: 'old'));
      expect(
        () => assets.registerPerCall({
          'lut': StaticAsset.text(name: 'lut', data: 'new'),
        }),
        throwsStateError,
      );
      expect(assets.get('lut')!.text, 'old');

      // Once nothing holds it, a per-call asset replaces it.
      await assets.release('lut');
      final token = assets.registerPerCall({
        'lut': StaticAsset.text(name: 'lut', data: 'new'),
      });
      expect(assets.get('lut')!.text, 'new');
      await token.dispose();
      expect(assets.get('lut'), isNull);
    });

    test('a per-call token drops only the asset it cached', () async {
      final first = assets.registerPerCall({
        'f': StaticAsset.text(name: 'f', data: '1'),
      });
      assets.clearAsset('f');
      final second = assets.registerPerCall({
        'f': StaticAsset.text(name: 'f', data: '2'),
      });

      await first.dispose();
      expect(assets.get('f')!.text, '2');
      await second.dispose();
      await second.dispose();
      expect(assets.get('f'), isNull);
    });

    test(
      'preloadBatch skips optional failures and undoes a required one',
      () async {
        await expectLater(
          assets.preloadBatch([
            _text('ok', 'x', cache: false),
            AssetConfig(name: 'optional', source: _missingFile),
            AssetConfig(name: 'needed', source: _missingFile, required: true),
          ]),
          throwsA(isA<FileSystemException>()),
        );
        expect(assets.get('ok'), isNull);
        expect(events, [
          'loaded ok',
          'failed optional',
          'failed needed',
          'evicted ok',
        ]);
      },
    );
  });

  group('embedded sources', () {
    test('a data string is text unless the hint names a binary type', () async {
      final gain = await assets.load(
        AssetConfig(name: 'g', source: {'type': 'embedded', 'data': 'gain'}),
      );
      expect(gain.text, 'gain');
      expect(gain.bytes, isNull);

      final weights = await assets.load(
        AssetConfig(
          name: 'w',
          source: {'type': 'embedded', 'data': 'AQID'},
          typeHint: 'weights',
        ),
      );
      expect(weights.bytes, [1, 2, 3]);
    });

    test('a source without a type is embedded', () async {
      final asset = await assets.load(
        AssetConfig(name: 'e', source: {'text': 'hi'}),
      );
      expect(asset.text, 'hi');
    });

    test('malformed embedded data fails to load', () async {
      Future<StaticAsset> load(Map<String, dynamic> source) =>
          assets.load(AssetConfig(name: 'bad', source: source));

      await expectLater(
        load({'type': 'embedded', 'bytes': 'not base64!'}),
        throwsFormatException,
      );
      await expectLater(
        load({
          'type': 'embedded',
          'bytes': [1, 256],
        }),
        throwsFormatException,
      );
      await expectLater(
        load({'type': 'embedded', 'text': 3}),
        throwsFormatException,
      );
      await expectLater(load({'type': 'embedded'}), throwsFormatException);
      expect(assets.get('bad'), isNull);
    });

    test('an unknown source type fails to load', () async {
      await expectLater(
        assets.load(AssetConfig(name: 'u', source: {'type': 'ftp'})),
        throwsArgumentError,
      );
    });
  });

  group('Pipeline assets', () {
    test(
      'start() holds preload assets until stop() and skips the rest',
      () async {
        final pipeline = Pipeline(id: 'p', assetManager: assets)
          ..setSerializedAssets([
            _text('now', 'a', cache: false),
            AssetConfig(
              name: 'later',
              source: {'type': 'embedded', 'text': 'b'},
              strategy: 'lazy',
            ),
            AssetConfig(name: 'optional', source: _missingFile),
          ]);

        await pipeline.start();
        expect(assets.get('now')!.text, 'a');
        expect(assets.get('later'), isNull);
        expect(assets.get('optional'), isNull);

        await pipeline.stop();
        expect(assets.get('now'), isNull);
        await pipeline.dispose();
      },
    );

    test(
      'a required asset that fails stops start() and releases the others',
      () async {
        final pipeline = Pipeline(id: 'p', assetManager: assets)
          ..setSerializedAssets([
            _text('first', 'a', cache: false),
            AssetConfig(name: 'needed', source: _missingFile, required: true),
          ]);

        await expectLater(
          pipeline.start(),
          throwsA(isA<FileSystemException>()),
        );
        expect(assets.get('first'), isNull);
        await pipeline.dispose();
      },
    );

    test(
      'a stage skips an optional asset and releases its assets on stop()',
      () async {
        await assets.setCacheLimit(1);
        final pipeline = Pipeline(id: 'p', assetManager: assets)
          ..addStage(
            StageBuilder('s')
                .withAsset(
                  asset: StaticAsset(
                    name: 'gone',
                    type: AssetType.binary,
                    source: _missingFile,
                    isRequired: false,
                  ),
                )
                .withAsset(
                  asset: StaticAsset.text(name: 'here', data: 'xy'),
                )
                .executeCPU(
                  (inputs, ranks, params) async => {'output': inputs['input']!},
                  inputKey: 'input',
                  outputKey: 'output',
                )
                .build(),
          );

        await pipeline.start();
        expect(events, contains('failed gone'));
        // Over the 1-byte limit, but the stage holds it.
        expect(assets.get('here')!.text, 'xy');

        await pipeline.stop();
        expect(assets.get('here'), isNull);
        await pipeline.dispose();
      },
    );
  });

  test('the validator checks network assets', () {
    final report = PipelineValidator().validate(
      NetworkConfig.decode(r'''
{
  "schema_version": 1,
  "assets": [
    {"name": "a", "source": {"type": "embedded", "text": "x"}},
    {"name": "a", "source": {"type": "embedded", "text": "y"}},
    {"name": "b", "source": {"type": "embedded", "text": "x"}, "strategy": "eager"},
    {"name": "c", "source": {"type": "file"}, "required": true},
    {"name": "d", "source": {"type": "embedded", "bytes": "!"}}
  ],
  "pipelines": [
    {
      "id": "p",
      "stages": [
        {
          "stage_id": "s",
          "stage_kind": "DynamicStage",
          "assets": ["a", "e"],
          "operations": [
            {
              "op_kind": "tensor_op",
              "operation": "multiplyScalar",
              "input_keys": ["input"],
              "output_keys": ["output"],
              "arguments": {"args": [2.0]}
            }
          ]
        }
      ],
      "connections": []
    }
  ]
}
'''),
    );

    String issue(ValidationIssue i) => '${i.code} ${i.path}';
    expect(report.errors.map(issue), [
      'DUP_ASSET_NAME assets[1].name',
      'ASSET_STRATEGY_UNKNOWN assets[2].strategy',
      'ASSET_SOURCE_INVALID assets[3].source',
    ]);
    // An optional asset that cannot load only warns: start() skips it.
    expect(
      report.warnings.map(issue),
      containsAll([
        'ASSET_SOURCE_INVALID assets[4].source',
        'ASSET_UNDECLARED pipelines[0].stages[0].assets[1]',
      ]),
    );
  });
}

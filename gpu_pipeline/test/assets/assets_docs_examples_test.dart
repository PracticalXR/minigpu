import 'dart:convert';
import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:gpu_pipeline/src/pipeline_assets.dart';
import 'package:gpu_pipeline/src/serialization/pipeline_models.dart';

void main() {
  test('docs examples: stage-provided + serialized refs', () async {
    final m = AssetManager();

    // Stage-provided
    await m.preload(StaticAsset.text(name: 'greeting', data: 'hello'));
    expect(m.get('greeting')!.text, 'hello');

    // Serialized refs (embedded json + base64 bin)
    await m.preloadBatch([
      AssetConfig(
        name: 'cfg_json',
        source: {'type': 'embedded', 'data': '{"a":1}'},
        typeHint: 'json',
      ),
      AssetConfig(
        name: 'lut_bin',
        source: {
          'type': 'embedded',
          'data': base64.encode([1, 2, 3, 4]),
        },
      ),
    ]);

    final cfg = m.getPreloadedConfig('cfg_json')!;
    final lut = m.getPreloadedBinary('lut_bin')!;

    expect(cfg['a'], 1);
    expect(lut, equals(Uint8List.fromList([1, 2, 3, 4])));
  });
}

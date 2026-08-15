// Web compile smoke for gpu_ml: compiled by BOTH dart2js and dart2wasm in
// test/web_compile_smoke_test.dart.  Pulls the whole web-safe API surface
// (gpu_ml.dart -> gpu_tensor -> minigpu -> minigpu_web js_interop bindings)
// through the compiler so web breakage is caught at test time, not by users.
//
// It only needs to COMPILE everywhere; executing the GPU paths requires a
// browser with WebGPU.
import 'dart:typed_data';

import 'package:gpu_ml/gpu_ml.dart';

Future<void> main(List<String> args) async {
  // Parse a minimal synthetic GGUF header (runs fine under any runtime).
  final bytes = _tinyGguf();
  final f = GgufFile.parse(bytes);
  print('gguf v${f.version}, ${f.tensors.length} tensors, '
      'arch=${f.metadata['general.architecture']}');
  print('f16(1.5) bits=0x${floatToHalfBits(1.5).toRadixString(16)} '
      'roundtrip=${halfBitsToFloat(floatToHalfBits(1.5))}');

  // Keep the GPU paths reachable so the compilers build them.  Guarded by a
  // runtime flag that is never set during the compile smoke.
  if (args.contains('--gpu')) {
    final t = await Tensor.create([2, 2],
        data: Float32List.fromList([1, 2, 3, 4]));
    final s = await t.softmax();
    final norm = await t.rmsNorm(await Tensor.create([2],
        data: Float32List.fromList([1, 1])));
    final r = await norm.rope(headDim: 2, heads: 2, positionOffset: 0);
    final q = await QuantizedTensor.create(
      [1, 32],
      GgmlType.q8_0,
      Uint8List(34),
    );
    final d = await q.dequantize();
    final y = await q.matVec(await Tensor.create([32]));
    print([s.shape, r.shape, d.shape, y.shape]);
  }
}

Uint8List _tinyGguf() {
  final b = BytesBuilder();
  void u32(int v) =>
      b.add((ByteData(4)..setUint32(0, v, Endian.little)).buffer.asUint8List());
  void u64(int v) {
    u32(v & 0xFFFFFFFF);
    u32(v ~/ 0x100000000);
  }

  void str(String s) {
    final e = s.codeUnits;
    u64(e.length);
    b.add(e);
  }

  u32(0x46554747); // GGUF
  u32(3);
  u64(0); // tensor count
  u64(1); // kv count
  str('general.architecture');
  u32(8); // string
  str('smoke');
  while (b.length % 32 != 0) {
    b.addByte(0);
  }
  return b.takeBytes();
}

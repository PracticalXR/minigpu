import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:gpu_ml/gpu_ml.dart';
import 'package:test/test.dart';

/// Synthetic-GGUF round trip: an in-test GGUF v3 writer produces a file with
/// f32 / f16 / Q8_0 / Q4_0 tensors; the parser + GPU loaders + fused kernels
/// are verified against CPU dequantization references that use the exact
/// stored (f16-rounded) scales, so agreement is tight.

// ---------------------------------------------------------------------------
// Minimal GGUF v3 writer
// ---------------------------------------------------------------------------

class _GgufWriter {
  final BytesBuilder _b = BytesBuilder();

  void u8(int v) => _b.addByte(v & 0xFF);
  void u32(int v) {
    final d = ByteData(4)..setUint32(0, v, Endian.little);
    _b.add(d.buffer.asUint8List());
  }

  void u64(int v) {
    u32(v & 0xFFFFFFFF);
    u32(v ~/ 0x100000000);
  }

  void str(String s) {
    final bytes = utf8.encode(s);
    u64(bytes.length);
    _b.add(bytes);
  }

  void bytes(List<int> data) => _b.add(data);

  int get length => _b.length;
  Uint8List take() => _b.takeBytes();
}

class _TensorSpec {
  _TensorSpec(this.name, this.shape, this.type, this.data);
  final String name;
  final List<int> shape; // [rows, cols] outermost-first
  final int type;
  final Uint8List data;
  int offset = 0;
}

Uint8List writeGguf(Map<String, dynamic> metadata, List<_TensorSpec> tensors) {
  const alignment = 32;
  final w = _GgufWriter();
  w.u32(0x46554747); // GGUF
  w.u32(3); // version
  w.u64(tensors.length);
  w.u64(metadata.length);

  metadata.forEach((key, value) {
    w.str(key);
    if (value is int) {
      w.u32(4); // uint32
      w.u32(value);
    } else if (value is String) {
      w.u32(8);
      w.str(value);
    } else if (value is List<int>) {
      w.u32(9); // array
      w.u32(4); // of uint32
      w.u64(value.length);
      for (final v in value) {
        w.u32(v);
      }
    } else {
      throw Exception('writer: unsupported metadata type');
    }
  });

  // Lay out tensor data at 32-byte-aligned offsets.
  int dataCursor = 0;
  for (final t in tensors) {
    dataCursor = ((dataCursor + alignment - 1) ~/ alignment) * alignment;
    t.offset = dataCursor;
    dataCursor += t.data.length;
  }

  for (final t in tensors) {
    w.str(t.name);
    final ne = t.shape.reversed.toList(); // GGUF stores innermost first
    w.u32(ne.length);
    for (final d in ne) {
      w.u64(d);
    }
    w.u32(t.type);
    w.u64(t.offset);
  }

  // Pad header to alignment, then emit the data section.
  while (w.length % alignment != 0) {
    w.u8(0);
  }
  int cursor = 0;
  for (final t in tensors) {
    while (cursor < t.offset) {
      w.u8(0);
      cursor++;
    }
    w.bytes(t.data);
    cursor += t.data.length;
  }
  return w.take();
}

// ---------------------------------------------------------------------------
// CPU quantizers + dequant references (use the STORED f16-rounded scale)
// ---------------------------------------------------------------------------

({Uint8List packed, Float32List reference}) quantizeQ8_0(Float32List values) {
  assert(values.length % 32 == 0);
  final nb = values.length ~/ 32;
  final packed = Uint8List(nb * 34);
  final reference = Float32List(values.length);
  final bd = ByteData.sublistView(packed);
  for (int b = 0; b < nb; b++) {
    double amax = 0;
    for (int l = 0; l < 32; l++) {
      amax = math.max(amax, values[b * 32 + l].abs());
    }
    final dBits = floatToHalfBits(amax / 127.0);
    final d = halfBitsToFloat(dBits);
    bd.setUint16(b * 34, dBits, Endian.little);
    for (int l = 0; l < 32; l++) {
      int q = d == 0 ? 0 : (values[b * 32 + l] / d).round();
      q = q.clamp(-127, 127);
      bd.setInt8(b * 34 + 2 + l, q);
      reference[b * 32 + l] = (d * q).toDouble();
    }
  }
  return (packed: packed, reference: reference);
}

({Uint8List packed, Float32List reference}) quantizeQ4_0(Float32List values) {
  assert(values.length % 32 == 0);
  final nb = values.length ~/ 32;
  final packed = Uint8List(nb * 18);
  final reference = Float32List(values.length);
  final bd = ByteData.sublistView(packed);
  for (int b = 0; b < nb; b++) {
    // ggml Q4_0: d = (signed value with max |v|) / -8.
    double maxSigned = 0;
    for (int l = 0; l < 32; l++) {
      final v = values[b * 32 + l];
      if (v.abs() > maxSigned.abs()) maxSigned = v;
    }
    final dBits = floatToHalfBits(maxSigned / -8.0);
    final d = halfBitsToFloat(dBits);
    bd.setUint16(b * 18, dBits, Endian.little);
    final qs = List<int>.filled(32, 8);
    for (int l = 0; l < 32; l++) {
      int q = d == 0 ? 8 : ((values[b * 32 + l] / d) + 8).round();
      q = q.clamp(0, 15);
      qs[l] = q;
      reference[b * 32 + l] = (d * (q - 8)).toDouble();
    }
    for (int l = 0; l < 16; l++) {
      bd.setUint8(b * 18 + 2 + l, (qs[l] & 0xF) | (qs[l + 16] << 4));
    }
  }
  return (packed: packed, reference: reference);
}

({Uint8List packed, Float32List reference}) quantizeF16(Float32List values) {
  final packed = Uint8List(values.length * 2);
  final reference = Float32List(values.length);
  final bd = ByteData.sublistView(packed);
  for (int i = 0; i < values.length; i++) {
    final bits = floatToHalfBits(values[i]);
    bd.setUint16(i * 2, bits, Endian.little);
    reference[i] = halfBitsToFloat(bits);
  }
  return (packed: packed, reference: reference);
}

Float32List seeded(int n, int seed) {
  final rng = math.Random(seed);
  final out = Float32List(n);
  for (int i = 0; i < n; i++) {
    out[i] = rng.nextDouble() * 2 - 1;
  }
  return out;
}

Float32List cpuMatVec(Float32List w, Float32List x, int rows, int cols) {
  final y = Float32List(rows);
  for (int i = 0; i < rows; i++) {
    double sum = 0;
    for (int k = 0; k < cols; k++) {
      sum += w[i * cols + k] * x[k];
    }
    y[i] = sum;
  }
  return y;
}

void expectClose(Float32List actual, Float32List expected, double tol) {
  expect(actual.length, equals(expected.length));
  for (int i = 0; i < expected.length; i++) {
    expect(actual[i], closeTo(expected[i], tol),
        reason: 'mismatch at index $i');
  }
}

// ---------------------------------------------------------------------------

Future<void> main() async {
  // rows=5 (odd) so Q8_0 rows straddle u32 word boundaries (34B blocks).
  const rows = 5, cols = 64;
  final wF32 = seeded(rows * cols, 100);
  final wF16src = seeded(rows * cols, 101);
  final wQ8src = seeded(rows * cols, 102);
  final wQ4src = seeded(rows * cols, 103);

  final f16q = quantizeF16(wF16src);
  final q8 = quantizeQ8_0(wQ8src);
  final q4 = quantizeQ4_0(wQ4src);

  final fileBytes = writeGguf(
    {
      'general.architecture': 'test',
      'general.alignment': 32,
      'test.layers': 5,
      'test.dims': [rows, cols],
    },
    [
      _TensorSpec('w.f32', [rows, cols], GgmlType.f32,
          wF32.buffer.asUint8List(wF32.offsetInBytes, wF32.lengthInBytes)),
      _TensorSpec('w.f16', [rows, cols], GgmlType.f16, f16q.packed),
      _TensorSpec('w.q8', [rows, cols], GgmlType.q8_0, q8.packed),
      _TensorSpec('w.q4', [rows, cols], GgmlType.q4_0, q4.packed),
    ],
  );

  group('GGUF parser', () {
    test('header, metadata, tensor directory', () {
      final f = GgufFile.parse(fileBytes);
      expect(f.version, equals(3));
      expect(f.metadata['general.architecture'], equals('test'));
      expect(f.metadata['test.layers'], equals(5));
      expect(f.metadata['test.dims'], equals([rows, cols]));
      expect(f.tensors.length, equals(4));

      final q8i = f.tensor('w.q8')!;
      expect(q8i.type, equals(GgmlType.q8_0));
      expect(q8i.ne, equals([cols, rows])); // innermost first
      expect(q8i.shape, equals([rows, cols]));
      expect(q8i.byteSize, equals(rows * cols ~/ 32 * 34));
      expect(f.tensorBytes(q8i), equals(q8.packed));
    });

    test('rejects bad magic', () {
      final bad = Uint8List.fromList([1, 2, 3, 4, 0, 0, 0, 0]);
      expect(() => GgufFile.parse(bad), throwsA(isA<Exception>()));
    });
  });

  group('f16 round trip helpers', () {
    test('floatToHalfBits/halfBitsToFloat round known values', () {
      expect(halfBitsToFloat(floatToHalfBits(1.0)), equals(1.0));
      expect(halfBitsToFloat(floatToHalfBits(-2.5)), equals(-2.5));
      expect(halfBitsToFloat(floatToHalfBits(0.0)), equals(0.0));
      expect(halfBitsToFloat(floatToHalfBits(0.333)), closeTo(0.333, 3e-4));
      expect(halfBitsToFloat(floatToHalfBits(65504)), equals(65504));
    });
  });

  group('QuantizedTensor', () {
    test('f16 dequantize matches CPU reference', () async {
      final f = GgufFile.parse(fileBytes);
      final qt = await f.loadQuantized('w.f16');
      final t = await qt.dequantize();
      expect(t.shape, equals([rows, cols]));
      final actual = await t.getData() as Float32List;
      expectClose(actual, f16q.reference, 1e-6);
    });

    test('q8_0 dequantize matches CPU reference', () async {
      final f = GgufFile.parse(fileBytes);
      final qt = await f.loadQuantized('w.q8');
      final t = await qt.dequantize();
      final actual = await t.getData() as Float32List;
      expectClose(actual, q8.reference, 1e-6);
    });

    test('q4_0 dequantize matches CPU reference', () async {
      final f = GgufFile.parse(fileBytes);
      final qt = await f.loadQuantized('w.q4');
      final t = await qt.dequantize();
      final actual = await t.getData() as Float32List;
      expectClose(actual, q4.reference, 1e-6);
    });

    test('f16 fused matVec matches CPU reference', () async {
      final f = GgufFile.parse(fileBytes);
      final qt = await f.loadQuantized('w.f16');
      final xData = seeded(cols, 200);
      final x = await Tensor.create([cols], data: xData);
      final y = await qt.matVec(x);
      expect(y.shape, equals([rows]));
      final actual = await y.getData() as Float32List;
      expectClose(actual, cpuMatVec(f16q.reference, xData, rows, cols), 1e-3);
    });

    test('q8_0 fused matVec matches CPU reference', () async {
      final f = GgufFile.parse(fileBytes);
      final qt = await f.loadQuantized('w.q8');
      final xData = seeded(cols, 201);
      final x = await Tensor.create([cols], data: xData);
      final y = await qt.matVec(x);
      final actual = await y.getData() as Float32List;
      expectClose(actual, cpuMatVec(q8.reference, xData, rows, cols), 1e-3);
    });

    test('q4_0 fused matVec matches CPU reference', () async {
      final f = GgufFile.parse(fileBytes);
      final qt = await f.loadQuantized('w.q4');
      final xData = seeded(cols, 202);
      final x = await Tensor.create([cols], data: xData);
      final y = await qt.matVec(x);
      final actual = await y.getData() as Float32List;
      expectClose(actual, cpuMatVec(q4.reference, xData, rows, cols), 1e-3);
    });

    test('loadF32 round-trips raw float data', () async {
      final f = GgufFile.parse(fileBytes);
      final t = await f.loadF32('w.f32');
      expect(t.shape, equals([rows, cols]));
      final actual = await t.getData() as Float32List;
      expectClose(actual, wF32, 0.0);
    });
  });
}

@TestOn('windows || mac-os || linux')
@Timeout(Duration(minutes: 5))
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:gpu_ml/gpu_ml_io.dart';
import 'package:test/test.dart';

/// K-quant (Q5_K / Q6_K) validation against REAL model data, three ways:
///
/// 1. GPU dequantize == CPU decode of the same packed bytes (exact) — proves
///    the WGSL mirrors quant_cpu.dart.
/// 2. Fused matVec == CPU decode + dot (sampled rows) — proves the fused
///    kernel indexes identically to the dequant path.
/// 3. CROSS-FILE: the same logical tensor decoded from the Q5_K_P file and
///    the Q8_0 file must agree closely (both approximate the same f32
///    originals, independently encoded by llama.cpp's quantizer).  This is
///    the check a shared-bug CPU reference cannot provide: a misread block
///    layout would decode "self-consistently" but diverge wildly here.
const q5Path =
    r'C:\models\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_P.gguf';
const q8Path =
    r'C:\models\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf';

/// Rows to pull from big tensors (whole-row ranges are valid sub-tensors).
const sampleRows = 64;

Future<({Uint8List packed, List<int> shape, int type})> readRows(
  GgufStream s,
  String name,
  int maxRows,
) async {
  final info = s.tensor(name)!;
  final cols = info.ne[0];
  final rows = math.min(info.ne[1], maxRows);
  final traits = ggmlTypeTraits[info.type]!;
  final bytesPerRow = cols ~/ traits.blockSize * traits.typeSize;
  final packed =
      await s.readTensorBytes(info, byteOffset: 0, byteLength: rows * bytesPerRow);
  return (packed: packed, shape: [rows, cols], type: info.type);
}

void main() {
  final available = File(q5Path).existsSync() && File(q8Path).existsSync();
  final skip = available ? false : 'model files not present';

  Future<void> gpuVsCpu(String name, int expectType) async {
    final s = await GgufStream.open(q5Path);
    try {
      final t = await readRows(s, name, sampleRows);
      expect(t.type, equals(expectType));
      final n = t.shape[0] * t.shape[1];
      final cpu = dequantizeCpu(t.type, t.packed, n);

      final qt = await QuantizedTensor.create(t.shape, t.type, t.packed);
      final gpuT = await qt.dequantize();
      final gpu = await gpuT.getData() as Float32List;
      for (int i = 0; i < n; i += 251) {
        expect(gpu[i], closeTo(cpu[i], 1e-6), reason: 'element $i');
      }

      // Fused matVec vs CPU decode + dot with a seeded vector.
      final rng = math.Random(5);
      final xData = Float32List(t.shape[1]);
      for (int i = 0; i < xData.length; i++) {
        xData[i] = rng.nextDouble() * 2 - 1;
      }
      final x = await Tensor.create([t.shape[1]], data: xData);
      final y = await qt.matVec(x);
      final yData = await y.getData() as Float32List;
      for (int r = 0; r < t.shape[0]; r += 7) {
        double sum = 0;
        for (int c = 0; c < t.shape[1]; c++) {
          sum += cpu[r * t.shape[1] + c] * xData[c];
        }
        expect(yData[r], closeTo(sum, sum.abs() * 1e-3 + 1e-2),
            reason: 'row $r');
      }
    } finally {
      await s.close();
    }
  }

  group('K-quants vs real Q5_K_P file', () {
    test('Q5_K: GPU dequant == CPU decode; fused matVec correct', () async {
      // token_embd.weight is Q5_K in this file.
      await gpuVsCpu('token_embd.weight', GgmlType.q5K);
    }, skip: skip);

    test('Q6_K: GPU dequant == CPU decode; fused matVec correct', () async {
      // output.weight (lm_head) is Q6_K in this file.
      await gpuVsCpu('output.weight', GgmlType.q6K);
    }, skip: skip);
  });

  group('cross-file consistency (independent llama.cpp encodings)', () {
    Future<void> crossFile(String name) async {
      final s5 = await GgufStream.open(q5Path);
      final s8 = await GgufStream.open(q8Path);
      try {
        final t5 = await readRows(s5, name, sampleRows);
        final t8 = await readRows(s8, name, sampleRows);
        expect(t5.shape, equals(t8.shape));
        final n = t5.shape[0] * t5.shape[1];
        final a = dequantizeCpu(t5.type, t5.packed, n);
        final b = dequantizeCpu(t8.type, t8.packed, n);
        // Relative RMS difference dominated by the coarser quant's error —
        // small for correct decodes, catastrophic for layout misreads.
        double num = 0, den = 0;
        for (int i = 0; i < n; i++) {
          final d = a[i] - b[i];
          num += d * d;
          den += b[i] * b[i];
        }
        final relRms = math.sqrt(num / (den + 1e-12));
        expect(den, greaterThan(0));
        expect(relRms, lessThan(0.10),
            reason: '$name rel RMS $relRms — decode layout suspect');
        // And not suspiciously zero (different quants should differ a bit).
        expect(relRms, greaterThan(1e-5));
      } finally {
        await s5.close();
        await s8.close();
      }
    }

    test('Q5_K vs Q8_0 on token_embd.weight', () async {
      await crossFile('token_embd.weight');
    }, skip: skip);

    test('Q6_K vs Q8_0 on output.weight', () async {
      await crossFile('output.weight');
    }, skip: skip);
  });

  group('expert stacks', () {
    test('3D Q8_0 stack: per-expert matVec matches CPU', () async {
      // Synthetic 4-expert stack, quantized per contiguous block.
      const experts = 4, rows = 8, cols = 64;
      final rng = math.Random(9);
      final values = Float32List(experts * rows * cols);
      for (int i = 0; i < values.length; i++) {
        values[i] = rng.nextDouble() * 2 - 1;
      }
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
          final q =
              d == 0 ? 0 : (values[b * 32 + l] / d).round().clamp(-127, 127);
          bd.setInt8(b * 34 + 2 + l, q);
          reference[b * 32 + l] = (d * q).toDouble();
        }
      }

      final qt = await QuantizedTensor.create(
        [experts, rows, cols],
        GgmlType.q8_0,
        packed,
      );
      expect(qt.experts, equals(experts));

      final xData = Float32List(cols);
      for (int i = 0; i < cols; i++) {
        xData[i] = rng.nextDouble() * 2 - 1;
      }
      final x = await Tensor.create([cols], data: xData);

      for (final expert in [0, 2, 3]) {
        final y = await qt.matVec(x, expert: expert);
        final yData = await y.getData() as Float32List;
        for (int r = 0; r < rows; r++) {
          double sum = 0;
          for (int c = 0; c < cols; c++) {
            sum += reference[(expert * rows + r) * cols + c] * xData[c];
          }
          expect(yData[r], closeTo(sum, 1e-3),
              reason: 'expert $expert row $r');
        }
      }
    });

    test('real MoE expert: range-read ONE expert from ffn_gate_exps',
        () async {
      final s = await GgufStream.open(q8Path);
      try {
        // blk.0.ffn_gate_exps.weight ne=[2048, 512, 256] Q8_0.
        final info = s.tensor('blk.0.ffn_gate_exps.weight')!;
        expect(info.ne.length, equals(3));
        final cols = info.ne[0], rows = info.ne[1];
        final traits = ggmlTypeTraits[info.type]!;
        final bytesPerExpert =
            rows * cols ~/ traits.blockSize * traits.typeSize;

        const expert = 17;
        final packed = await s.readTensorBytes(
          info,
          byteOffset: expert * bytesPerExpert,
          byteLength: bytesPerExpert,
        );
        final cpu = dequantizeCpu(info.type, packed, rows * cols);

        final qt =
            await QuantizedTensor.create([rows, cols], info.type, packed);
        final xData = Float32List(cols)..fillRange(0, cols, 1.0);
        final x = await Tensor.create([cols], data: xData);
        final y = await qt.matVec(x);
        final yData = await y.getData() as Float32List;
        for (int r = 0; r < rows; r += 31) {
          double sum = 0;
          for (int c = 0; c < cols; c++) {
            sum += cpu[r * cols + c];
          }
          expect(yData[r], closeTo(sum, sum.abs() * 1e-3 + 1e-2),
              reason: 'row $r');
        }
      } finally {
        await s.close();
      }
    }, skip: File(q8Path).existsSync() ? false : 'model file not present');
  });
}

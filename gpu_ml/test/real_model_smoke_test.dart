@TestOn('windows || mac-os || linux')
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:gpu_ml/gpu_ml_io.dart';
import 'package:test/test.dart';

/// Streams REAL tensors out of the local Qwen3.6 GGUF (header parse + range
/// reads only — the 40 GB file is never loaded) and pushes them through the
/// GPU quant kernels.  Skipped when the model file is absent so CI stays
/// green on machines without C:\models.
const modelPath =
    r'C:\models\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf';

Future<void> main() async {
  final available = File(modelPath).existsSync();

  group('real Qwen3.6 GGUF', () {
    test('header parses with expected architecture', () async {
      final s = await GgufStream.open(modelPath);
      try {
        expect(s.header.version, equals(3));
        expect(s.metadata['general.architecture'], equals('qwen35moe'));
        expect(s.metadata['qwen35moe.block_count'], equals(40));
        expect(s.metadata['qwen35moe.expert_count'], equals(256));
        expect(s.tensors.length, equals(733));
        expect(s.tensor('token_embd.weight'), isNotNull);
        expect(s.tensor('blk.0.ssm_out.weight'), isNotNull);
        expect(s.tensor('blk.3.attn_q.weight'), isNotNull);
      } finally {
        await s.close();
      }
    }, skip: available ? false : 'model file not present');

    test('f32 norm tensor loads with sane values', () async {
      final s = await GgufStream.open(modelPath);
      try {
        final t = await s.loadF32('blk.0.attn_norm.weight');
        expect(t.shape, equals([2048]));
        final data = await t.getData() as Float32List;
        // RMSNorm gains cluster around O(1); all finite, not all zero.
        double maxAbs = 0, sumAbs = 0;
        for (final v in data) {
          expect(v.isFinite, isTrue);
          maxAbs = math.max(maxAbs, v.abs());
          sumAbs += v.abs();
        }
        expect(maxAbs, greaterThan(0));
        expect(maxAbs, lessThan(1000));
        expect(sumAbs / data.length, greaterThan(1e-4));
      } finally {
        await s.close();
      }
    }, skip: available ? false : 'model file not present');

    test('Q8_0 weight streams to VRAM; GPU dequant matches CPU dequant',
        () async {
      final s = await GgufStream.open(modelPath);
      try {
        // Shared-expert gate projection: [512, 2048] Q8_0, ~1.1 MB.
        final info = s.tensor('blk.0.ffn_gate_shexp.weight')!;
        expect(info.type, equals(GgmlType.q8_0));
        final packed = await s.readTensorBytes(info);
        expect(packed.length, equals(info.byteSize));

        // CPU dequant reference straight from the packed blocks.
        final n = info.elementCount;
        final cpu = Float32List(n);
        final bd = ByteData.sublistView(packed);
        for (int b = 0; b < n ~/ 32; b++) {
          final d = halfBitsToFloat(bd.getUint16(b * 34, Endian.little));
          for (int l = 0; l < 32; l++) {
            cpu[b * 32 + l] = d * bd.getInt8(b * 34 + 2 + l);
          }
        }

        final qt = await QuantizedTensor.create(
          info.shape,
          info.type,
          packed,
        );
        final t = await qt.dequantize();
        final gpuData = await t.getData() as Float32List;
        expect(gpuData.length, equals(n));
        for (int i = 0; i < n; i += 997) {
          // stride-sample the 1M elements
          expect(gpuData[i], closeTo(cpu[i], 1e-6),
              reason: 'mismatch at element $i');
        }

        // Fused matVec against ones == row sums of the dequant reference.
        final rows = info.shape[0], cols = info.shape[1];
        final ones = await Tensor.create(
          [cols],
          data: Float32List(cols)..fillRange(0, cols, 1.0),
        );
        final y = await qt.matVec(ones);
        final yData = await y.getData() as Float32List;
        for (int r = 0; r < rows; r += 37) {
          double sum = 0;
          for (int c = 0; c < cols; c++) {
            sum += cpu[r * cols + c];
          }
          expect(yData[r], closeTo(sum, sum.abs() * 1e-3 + 1e-2),
              reason: 'row-sum mismatch at row $r');
        }
      } finally {
        await s.close();
      }
    }, skip: available ? false : 'model file not present');

    test('F16 attention weight streams and dequantizes finite', () async {
      final s = await GgufStream.open(modelPath);
      try {
        final info = s.tensor('blk.0.ssm_alpha.weight')!;
        expect(info.type, equals(GgmlType.f16));
        final qt = await s.loadQuantized('blk.0.ssm_alpha.weight');
        final t = await qt.dequantize();
        final data = await t.getData() as Float32List;
        double maxAbs = 0;
        for (final v in data) {
          expect(v.isFinite, isTrue);
          maxAbs = math.max(maxAbs, v.abs());
        }
        expect(maxAbs, greaterThan(0));
      } finally {
        await s.close();
      }
    }, skip: available ? false : 'model file not present');
  });
}

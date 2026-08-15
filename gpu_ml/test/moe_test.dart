@TestOn('windows || mac-os || linux')
@Timeout(Duration(minutes: 5))
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:gpu_ml/gpu_ml_io.dart';
import 'package:test/test.dart';

import 'quant_test_utils.dart';

const q8Path =
    r'C:\models\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf';

void main() {
  group('MoeFfn synthetic', () {
    test('forward matches CPU reference (4 experts, top-2, shared)', () async {
      const experts = 4, dim = 64, ff = 32, topK = 2;

      final routerData = seeded(experts * dim, 30);
      final gateVals = seeded(experts * ff * dim, 31);
      final upVals = seeded(experts * ff * dim, 32);
      final downVals = seeded(experts * dim * ff, 33);
      final gateShVals = seeded(ff * dim, 34);
      final upShVals = seeded(ff * dim, 35);
      final downShVals = seeded(dim * ff, 36);
      final sharedGateData = seeded(dim, 37);
      final xData = seeded(dim, 38);

      final gateQ = quantizeQ8_0(gateVals);
      final upQ = quantizeQ8_0(upVals);
      final downQ = quantizeQ8_0(downVals);
      final gateShQ = quantizeQ8_0(gateShVals);
      final upShQ = quantizeQ8_0(upShVals);
      final downShQ = quantizeQ8_0(downShVals);

      final moe = MoeFfn(
        router: await Tensor.create([experts, dim], data: routerData),
        gateExps: await QuantizedTensor.create(
            [experts, ff, dim], GgmlType.q8_0, gateQ.packed),
        upExps: await QuantizedTensor.create(
            [experts, ff, dim], GgmlType.q8_0, upQ.packed),
        downExps: await QuantizedTensor.create(
            [experts, dim, ff], GgmlType.q8_0, downQ.packed),
        topK: topK,
        gateShexp: await QuantizedTensor.create(
            [ff, dim], GgmlType.q8_0, gateShQ.packed),
        upShexp: await QuantizedTensor.create(
            [ff, dim], GgmlType.q8_0, upShQ.packed),
        downShexp: await QuantizedTensor.create(
            [dim, ff], GgmlType.q8_0, downShQ.packed),
        sharedGate: await Tensor.create([dim], data: sharedGateData),
      );

      final x = List<double>.generate(dim, (i) => xData[i].toDouble());

      // CPU routing.
      final logits = cpuMatVec(routerData, x, experts, dim);
      final mx = logits.reduce(math.max);
      final exps = logits.map((v) => math.exp(v - mx)).toList();
      final sum = exps.reduce((a, b) => a + b);
      final probs = exps.map((v) => v / sum).toList();
      final order = List<int>.generate(experts, (i) => i)
        ..sort((a, b) => probs[b].compareTo(probs[a]));
      final sel = order.take(topK).toList();
      final wSum = sel.fold(0.0, (a, i) => a + probs[i]);

      // GPU routing must agree.
      final xT = await Tensor.create([dim], data: xData);
      final routing = await moe.route(xT);
      expect(routing.indices.toSet(), equals(sel.toSet()));
      for (int k = 0; k < topK; k++) {
        final i = routing.indices[k];
        expect(routing.weights[k], closeTo(probs[i] / wSum, 1e-4));
      }

      // CPU forward with the effective (quant-reference) weights.
      Float32List slice(Float32List v, int e, int r, int c) =>
          Float32List.sublistView(v, e * r * c, (e + 1) * r * c);
      final expected = List<double>.filled(dim, 0);
      for (final e in sel) {
        final out = cpuExpertFfn(slice(gateQ.reference, e, ff, dim),
            slice(upQ.reference, e, ff, dim),
            slice(downQ.reference, e, dim, ff), x, ff, dim);
        for (int i = 0; i < dim; i++) {
          expected[i] += (probs[e] / wSum) * out[i];
        }
      }
      double dot = 0;
      for (int i = 0; i < dim; i++) {
        dot += sharedGateData[i] * x[i];
      }
      final gVal = 1.0 / (1.0 + math.exp(-dot));
      final shOut = cpuExpertFfn(gateShQ.reference, upShQ.reference,
          downShQ.reference, x, ff, dim);
      for (int i = 0; i < dim; i++) {
        expected[i] += gVal * shOut[i];
      }

      final outT = await moe.forward(xT);
      final actual = await outT.getData() as Float32List;
      for (int i = 0; i < dim; i++) {
        expect(actual[i], closeTo(expected[i], 1e-3), reason: 'element $i');
      }
      moe.destroy();
    });
  });

  group('MoeFfn real blk.0 (Q8_K_P file)', () {
    test('full layer forward matches CPU decode reference', () async {
      final s = await GgufStream.open(q8Path);
      try {
        final moe = await s.loadMoeFfn(0);
        expect(moe.experts, equals(256));
        expect(moe.topK, equals(8));
        expect(moe.dim, equals(2048));
        expect(moe.ff, equals(512));

        final xData = seeded(moe.dim, 40);
        final x = List<double>.generate(moe.dim, (i) => xData[i].toDouble());
        final xT = await Tensor.create([moe.dim], data: xData);

        final routing = await moe.route(xT);
        expect(routing.indices.length, equals(8));
        final outT = await moe.forward(xT);
        final actual = await outT.getData() as Float32List;

        // CPU reference: decode ONLY the routed experts straight from disk
        // (range reads — independent of the GPU upload path).
        final gi = s.tensor('blk.0.ffn_gate_exps.weight')!;
        final ui = s.tensor('blk.0.ffn_up_exps.weight')!;
        final di = s.tensor('blk.0.ffn_down_exps.weight')!;
        final ffDim = moe.ff, dim = moe.dim;
        Future<Float32List> expertSlice(
            GgufTensorInfo info, int e, int r, int c) async {
          final traits = ggmlTypeTraits[info.type]!;
          final bytesPer = r * c ~/ traits.blockSize * traits.typeSize;
          final packed = await s.readTensorBytes(info,
              byteOffset: e * bytesPer, byteLength: bytesPer);
          return dequantizeCpu(info.type, packed, r * c);
        }

        final expected = List<double>.filled(dim, 0);
        for (int k = 0; k < routing.indices.length; k++) {
          final e = routing.indices[k];
          final w = routing.weights[k];
          final out = cpuExpertFfn(
            await expertSlice(gi, e, ffDim, dim),
            await expertSlice(ui, e, ffDim, dim),
            await expertSlice(di, e, dim, ffDim),
            x,
            ffDim,
            dim,
          );
          for (int i = 0; i < dim; i++) {
            expected[i] += w * out[i];
          }
        }

        // Shared expert.
        final sharedGate =
            await s.loadF32('blk.0.ffn_gate_inp_shexp.weight');
        final sgData = await sharedGate.getData() as Float32List;
        double dot = 0;
        for (int i = 0; i < dim; i++) {
          dot += sgData[i] * x[i];
        }
        final gVal = 1.0 / (1.0 + math.exp(-dot));
        Future<Float32List> shexp(String name, int r, int c) async {
          final info = s.tensor('blk.0.$name')!;
          return dequantizeCpu(
              info.type, await s.readTensorBytes(info), r * c);
        }

        final shOut = cpuExpertFfn(
          await shexp('ffn_gate_shexp.weight', ffDim, dim),
          await shexp('ffn_up_shexp.weight', ffDim, dim),
          await shexp('ffn_down_shexp.weight', dim, ffDim),
          x,
          ffDim,
          dim,
        );
        for (int i = 0; i < dim; i++) {
          expected[i] += gVal * shOut[i];
        }

        double maxAbsErr = 0, maxMag = 0;
        for (int i = 0; i < dim; i++) {
          maxAbsErr = math.max(maxAbsErr, (actual[i] - expected[i]).abs());
          maxMag = math.max(maxMag, expected[i].abs());
        }
        expect(maxMag, greaterThan(0));
        expect(maxAbsErr, lessThan(math.max(1e-3, maxMag * 1e-3)),
            reason: 'maxAbsErr $maxAbsErr vs maxMag $maxMag');

        moe.destroy();
      } finally {
        await s.close();
      }
    }, skip: File(q8Path).existsSync() ? false : 'model file not present');
  });
}

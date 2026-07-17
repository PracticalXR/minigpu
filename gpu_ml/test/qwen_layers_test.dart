@TestOn('windows || mac-os || linux')
@Timeout(Duration(minutes: 10))
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:gpu_ml/gpu_ml_io.dart';
import 'package:test/test.dart';

import 'quant_test_utils.dart';

/// qwen35moe layer modules vs CPU references that mirror
/// docs/QWEN35MOE_SEMANTICS.md exactly, using REAL weights from the Q8_K_P
/// file.  A relRms near 1.0 means zeros (binding/validation failure); small
/// relRms means numerics agree.
const q8Path =
    r'C:\models\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf';

double relRms(Float32List a, List<double> b) {
  double num = 0, den = 0;
  for (int i = 0; i < a.length; i++) {
    final d = a[i] - b[i];
    num += d * d;
    den += b[i] * b[i];
  }
  return math.sqrt(num / (den + 1e-30));
}

double softplus(double x) => x > 20 ? x : math.log(1 + math.exp(x));
double sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));
double siluD(double x) => x * sigmoid(x);

List<double> rmsNormCpu(List<double> x, Float32List w, double eps) {
  double ss = 0;
  for (final v in x) {
    ss += v * v;
  }
  final inv = 1.0 / math.sqrt(ss / x.length + eps);
  return List<double>.generate(x.length, (j) => x[j] * inv * w[j]);
}

/// NEOX partial rope on one [headDim] head.
void ropeNeoxCpu(List<double> h, int rot, int pos, double base) {
  final half = rot ~/ 2;
  for (int i = 0; i < half; i++) {
    final angle = pos * math.pow(base, -2.0 * i / rot).toDouble();
    final c = math.cos(angle), s = math.sin(angle);
    final x0 = h[i], x1 = h[i + half];
    h[i] = x0 * c - x1 * s;
    h[i + half] = x0 * s + x1 * c;
  }
}

Future<Float32List> decode(GgufStream s, String name) async {
  final info = s.tensor(name)!;
  return dequantizeCpu(
      info.type, await s.readTensorBytes(info), info.elementCount);
}

void main() {
  final available = File(q8Path).existsSync();
  final skip = available ? false : 'model file not present';

  group('ropeNeox / l2NormRows units', () {
    test('ropeNeox partial matches CPU', () async {
      const heads = 2, headDim = 16, rot = 8, pos = 5;
      final data = seeded(heads * headDim, 50);
      final t = await Tensor.create([heads, headDim], data: data);
      final r = await t.ropeNeox(
          headDim: headDim,
          heads: heads,
          positionOffset: pos,
          ropeDims: rot,
          thetaBase: 100.0);
      final actual = await r.getData() as Float32List;
      final expected = <double>[];
      for (int h = 0; h < heads; h++) {
        final row = List<double>.generate(
            headDim, (i) => data[h * headDim + i].toDouble());
        ropeNeoxCpu(row, rot, pos, 100.0);
        expected.addAll(row);
      }
      for (int i = 0; i < expected.length; i++) {
        expect(actual[i], closeTo(expected[i], 1e-4), reason: 'elem $i');
      }
    });

    test('l2NormRows matches CPU (incl. eps floor on zero row)', () async {
      const rows = 3, d = 64;
      final data = seeded(rows * d, 51);
      for (int j = 0; j < d; j++) {
        data[2 * d + j] = 0; // zero row exercises the eps floor
      }
      final t = await Tensor.create([rows, d], data: data);
      final r = await t.l2NormRows(eps: 1e-6);
      final actual = await r.getData() as Float32List;
      for (int rIdx = 0; rIdx < rows; rIdx++) {
        double ss = 0;
        for (int j = 0; j < d; j++) {
          ss += data[rIdx * d + j] * data[rIdx * d + j];
        }
        final inv = 1.0 / math.max(math.sqrt(ss), 1e-6);
        for (int j = 0; j < d; j++) {
          expect(actual[rIdx * d + j], closeTo(data[rIdx * d + j] * inv, 1e-5),
              reason: 'row $rIdx elem $j');
        }
      }
    });
  });

  group('real blk.3 attention layer', () {
    test('decode step matches CPU reference', () async {
      final s = await GgufStream.open(q8Path);
      try {
        final layer = await s.loadAttentionLayer(3);
        const heads = 16, kvHeads = 2, headDim = 256, rot = 64;
        const dim = 2048, past = 3, pos = past, seqLen = past + 1;
        const base = 10000000.0;
        const eps = 1e-6;

        final xn = seeded(dim, 60);
        final pastK = seeded(past * kvHeads * headDim, 61);
        final pastV = seeded(past * kvHeads * headDim, 62);

        // ---- CPU reference ----
        final wqW = await decode(s, 'blk.3.attn_q.weight');
        final wkW = await decode(s, 'blk.3.attn_k.weight');
        final wvW = await decode(s, 'blk.3.attn_v.weight');
        final woW = await decode(s, 'blk.3.attn_output.weight');
        final qNormW = await decode(s, 'blk.3.attn_q_norm.weight');
        final kNormW = await decode(s, 'blk.3.attn_k_norm.weight');

        final x = List<double>.generate(dim, (i) => xn[i].toDouble());
        final qFull = cpuMatVec(wqW, x, heads * headDim * 2, dim);
        final kProj = cpuMatVec(wkW, x, kvHeads * headDim, dim);
        final vProj = cpuMatVec(wvW, x, kvHeads * headDim, dim);

        final qHeads = <List<double>>[];
        final gates = <double>[];
        for (int h = 0; h < heads; h++) {
          final qh = qFull.sublist(h * 512, h * 512 + 256);
          gates.addAll(qFull.sublist(h * 512 + 256, h * 512 + 512));
          final qn = rmsNormCpu(qh, qNormW, eps);
          ropeNeoxCpu(qn, rot, pos, base);
          qHeads.add(qn);
        }
        final kHeadsList = <List<double>>[];
        for (int h = 0; h < kvHeads; h++) {
          final kh = kProj.sublist(h * headDim, (h + 1) * headDim);
          final kn = rmsNormCpu(kh, kNormW, eps);
          ropeNeoxCpu(kn, rot, pos, base);
          kHeadsList.add(kn);
        }

        // Cache: past rows then current, [seq][kvHeads][headDim].
        double kAt(int t, int kv, int j) => t < past
            ? pastK[(t * kvHeads + kv) * headDim + j].toDouble()
            : kHeadsList[kv][j];
        double vAt(int t, int kv, int j) => t < past
            ? pastV[(t * kvHeads + kv) * headDim + j].toDouble()
            : vProj[kv * headDim + j];

        final attnFlat = List<double>.filled(heads * headDim, 0);
        for (int h = 0; h < heads; h++) {
          final kv = h ~/ (heads ~/ kvHeads);
          final scores = List<double>.generate(seqLen, (t) {
            double sc = 0;
            for (int j = 0; j < headDim; j++) {
              sc += qHeads[h][j] * kAt(t, kv, j);
            }
            return sc / math.sqrt(headDim);
          });
          final mx = scores.reduce(math.max);
          final exps = scores.map((v) => math.exp(v - mx)).toList();
          final sum = exps.reduce((a, b) => a + b);
          for (int t = 0; t < seqLen; t++) {
            final p = exps[t] / sum;
            for (int j = 0; j < headDim; j++) {
              attnFlat[h * headDim + j] += p * vAt(t, kv, j);
            }
          }
        }
        for (int i = 0; i < attnFlat.length; i++) {
          attnFlat[i] *= sigmoid(gates[i]);
        }
        final expected = cpuMatVec(woW, attnFlat, dim, heads * headDim);

        // ---- GPU ----
        final xT = await Tensor.create([dim], data: xn);
        final proj = await layer.project(xT, pos);
        final kCur = await proj.k.getData() as Float32List;
        final vCur = await proj.v.getData() as Float32List;
        final kAllData = Float32List(seqLen * kvHeads * headDim)
          ..setRange(0, past * kvHeads * headDim, pastK)
          ..setRange(past * kvHeads * headDim, seqLen * kvHeads * headDim, kCur);
        final vAllData = Float32List(seqLen * kvHeads * headDim)
          ..setRange(0, past * kvHeads * headDim, pastV)
          ..setRange(past * kvHeads * headDim, seqLen * kvHeads * headDim, vCur);
        final kAll =
            await Tensor.create([seqLen, kvHeads, headDim], data: kAllData);
        final vAll =
            await Tensor.create([seqLen, kvHeads, headDim], data: vAllData);
        final outT = await layer.attend(
            q: proj.q, gate: proj.gate, kAll: kAll, vAll: vAll);
        final actual = await outT.getData() as Float32List;

        final r = relRms(actual, expected);
        expect(r, lessThan(2e-3), reason: 'attention relRms $r');
        layer.destroy();
      } finally {
        await s.close();
      }
    }, skip: skip);
  });

  group('real blk.0 DeltaNet layer', () {
    test('two decode steps match CPU reference (state evolves)', () async {
      final s = await GgufStream.open(q8Path);
      try {
        final layer = await s.loadDeltaNetLayer(0);
        const dim = 2048, kH = 16, vH = 32, hd = 128, kernel = 4;
        const keyDim = kH * hd, valueDim = vH * hd;
        const convDim = 2 * keyDim + valueDim;
        const eps = 1e-6;

        // ---- CPU weights ----
        final wqkvW = await decode(s, 'blk.0.attn_qkv.weight');
        final wzW = await decode(s, 'blk.0.attn_gate.weight');
        final wbW = await decode(s, 'blk.0.ssm_beta.weight');
        final waW = await decode(s, 'blk.0.ssm_alpha.weight');
        final convW = await decode(s, 'blk.0.ssm_conv1d.weight');
        final aW = await decode(s, 'blk.0.ssm_a');
        final dtW = await decode(s, 'blk.0.ssm_dt.bias');
        final normW = await decode(s, 'blk.0.ssm_norm.weight');
        final woW = await decode(s, 'blk.0.ssm_out.weight');

        // ---- CPU state ----
        final hist =
            List.generate(kernel - 1, (_) => List<double>.filled(convDim, 0));
        final S = List.generate(
            vH, (_) => List.generate(hd, (_) => List<double>.filled(hd, 0)));

        List<double> cpuStep(Float32List xnF) {
          final x = List<double>.generate(dim, (i) => xnF[i].toDouble());
          final qkv = cpuMatVec(wqkvW, x, convDim, dim);
          final z = cpuMatVec(wzW, x, valueDim, dim);
          final betaRaw = cpuMatVec(wbW, x, vH, dim);
          final alpha = cpuMatVec(waW, x, vH, dim);

          // conv + silu + roll history.
          final conv = List<double>.filled(convDim, 0);
          for (int c = 0; c < convDim; c++) {
            double acc = convW[c * kernel + kernel - 1] * qkv[c];
            for (int t = 0; t < kernel - 1; t++) {
              acc += convW[c * kernel + t] * hist[t][c];
            }
            conv[c] = siluD(acc);
          }
          for (int t = 0; t + 1 < kernel - 1; t++) {
            hist[t] = hist[t + 1];
          }
          hist[kernel - 2] = qkv.sublist(0, convDim);

          // split + L2 norm.
          List<double> l2(List<double> v) {
            double ss = 0;
            for (final e in v) {
              ss += e * e;
            }
            final inv = 1.0 / math.max(math.sqrt(ss), eps);
            return v.map((e) => e * inv).toList();
          }

          final qN = List.generate(
              kH, (h) => l2(conv.sublist(h * hd, (h + 1) * hd)));
          final kN = List.generate(kH,
              (h) => l2(conv.sublist(keyDim + h * hd, keyDim + (h + 1) * hd)));

          final out = List<double>.filled(valueDim, 0);
          for (int h = 0; h < vH; h++) {
            final kk = h % kH; // tile mapping (ggml_repeat)
            final g = aW[h] * softplus(alpha[h] + dtW[h]);
            final decay = math.exp(g);
            final beta = sigmoid(betaRaw[h]);
            final q = qN[kk].map((e) => e / math.sqrt(hd)).toList();
            final k = kN[kk];
            final v =
                conv.sublist(2 * keyDim + h * hd, 2 * keyDim + (h + 1) * hd);
            // decay + S^T k
            final sk = List<double>.filled(hd, 0);
            for (int i = 0; i < hd; i++) {
              for (int j = 0; j < hd; j++) {
                S[h][i][j] *= decay;
                sk[j] += S[h][i][j] * k[i];
              }
            }
            for (int j = 0; j < hd; j++) {
              final d = (v[j] - sk[j]) * beta;
              for (int i = 0; i < hd; i++) {
                S[h][i][j] += k[i] * d;
              }
            }
            for (int j = 0; j < hd; j++) {
              double o = 0;
              for (int i = 0; i < hd; i++) {
                o += S[h][i][j] * q[i];
              }
              out[h * hd + j] = o;
            }
          }

          // gated norm + out proj.
          final gated = List<double>.filled(valueDim, 0);
          for (int h = 0; h < vH; h++) {
            final oh = rmsNormCpu(out.sublist(h * hd, (h + 1) * hd), normW, eps);
            for (int j = 0; j < hd; j++) {
              gated[h * hd + j] = oh[j] * siluD(z[h * hd + j]);
            }
          }
          return cpuMatVec(woW, gated, dim, valueDim);
        }

        // ---- run two tokens through both ----
        for (int step = 0; step < 2; step++) {
          final xn = seeded(dim, 70 + step);
          final expected = cpuStep(xn);
          final xT = await Tensor.create([dim], data: xn);
          final outT = await layer.forward(xT);
          final actual = await outT.getData() as Float32List;
          final r = relRms(actual, expected);
          expect(r, lessThan(2e-3), reason: 'step $step relRms $r');
          xT.destroy();
          outT.destroy();
        }
        layer.destroy();
      } finally {
        await s.close();
      }
    }, skip: skip);
  });
}

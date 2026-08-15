import 'dart:math' as math;
import 'dart:typed_data';

import 'package:gpu_ml/gpu_ml.dart';
import 'package:test/test.dart';

/// End-to-end llama decoder block, single-token decode step, computed twice:
/// once on CPU in doubles (the reference) and once as a composition of the
/// library's GPU ops — Q8_0 fused matVec projections, rmsNorm, rope, batched
/// per-head attention via reshape/transpose/matMul/softmax, SiLU-gated FFN.
///
/// Both sides use the SAME effective weights (the Q8_0 dequant reference,
/// i.e. stored-f16-scale * int8), so disagreement means a kernel bug, not
/// quantization noise.
///
/// The only CPU step in the GPU path is KV-cache assembly (readback of the
/// current k/v + concat with past rows + re-upload) — a GPU cache-append op
/// is a listed Phase 4 gap.

const dim = 64;
const heads = 4;
const headDim = 16;
const hidden = 96; // FFN width, multiple of 32 for Q8_0 blocks
const past = 3; // tokens already in the KV cache
const pos = past; // position of the token being decoded
const seqLen = past + 1;
const eps = 1e-5;

Float32List seeded(int n, int seed) {
  final rng = math.Random(seed);
  final out = Float32List(n);
  for (int i = 0; i < n; i++) {
    out[i] = rng.nextDouble() * 2 - 1;
  }
  return out;
}

/// Q8_0 quantize returning packed bytes + the effective (reference) weights.
({Uint8List packed, Float32List reference}) quantizeQ8_0(Float32List values) {
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
      int q = d == 0 ? 0 : (values[b * 32 + l] / d).round().clamp(-127, 127);
      bd.setInt8(b * 34 + 2 + l, q);
      reference[b * 32 + l] = (d * q).toDouble();
    }
  }
  return (packed: packed, reference: reference);
}

// --------------------------- CPU reference ---------------------------------

List<double> matVecCpu(Float32List w, List<double> x, int rows, int cols) {
  final y = List<double>.filled(rows, 0);
  for (int i = 0; i < rows; i++) {
    double s = 0;
    for (int k = 0; k < cols; k++) {
      s += w[i * cols + k] * x[k];
    }
    y[i] = s;
  }
  return y;
}

List<double> rmsNormCpu(List<double> x, Float32List w) {
  double ss = 0;
  for (final v in x) {
    ss += v * v;
  }
  final inv = 1.0 / math.sqrt(ss / x.length + eps);
  return List<double>.generate(x.length, (j) => x[j] * inv * w[j]);
}

List<double> ropeCpu(List<double> x, int position) {
  final out = List<double>.from(x);
  for (int h = 0; h < heads; h++) {
    for (int i = 0; i < headDim ~/ 2; i++) {
      final angle =
          position * math.pow(10000.0, -2.0 * i / headDim).toDouble();
      final c = math.cos(angle), s = math.sin(angle);
      final b = h * headDim + 2 * i;
      final x0 = out[b], x1 = out[b + 1];
      out[b] = x0 * c - x1 * s;
      out[b + 1] = x0 * s + x1 * c;
    }
  }
  return out;
}

Future<void> main() async {
  test('llama decoder block: GPU composition matches CPU reference', () async {
    // ---- weights & inputs -------------------------------------------------
    final wqQ = quantizeQ8_0(seeded(dim * dim, 10));
    final wkQ = quantizeQ8_0(seeded(dim * dim, 11));
    final wvQ = quantizeQ8_0(seeded(dim * dim, 12));
    final woQ = quantizeQ8_0(seeded(dim * dim, 13));
    final w1Q = quantizeQ8_0(seeded(hidden * dim, 14)); // gate
    final w3Q = quantizeQ8_0(seeded(hidden * dim, 15)); // up
    final w2Q = quantizeQ8_0(seeded(dim * hidden, 16)); // down

    final rms1 = seeded(dim, 17);
    final rms2 = seeded(dim, 18);
    final xIn = seeded(dim, 19);
    // Past K/V rows (already roped, as a real cache would hold them).
    final pastK = seeded(past * dim, 20);
    final pastV = seeded(past * dim, 21);

    // ---- CPU reference ----------------------------------------------------
    final x = xIn.map((v) => v.toDouble()).toList();
    final xn1 = rmsNormCpu(x, rms1);
    final q = ropeCpu(matVecCpu(wqQ.reference, xn1, dim, dim), pos);
    final k = ropeCpu(matVecCpu(wkQ.reference, xn1, dim, dim), pos);
    final v = matVecCpu(wvQ.reference, xn1, dim, dim);

    // K/V caches: past rows then the current token's row.
    final kc = List<double>.generate(
      seqLen * dim,
      (i) => i < past * dim ? pastK[i].toDouble() : k[i - past * dim],
    );
    final vc = List<double>.generate(
      seqLen * dim,
      (i) => i < past * dim ? pastV[i].toDouble() : v[i - past * dim],
    );

    final attnFlat = List<double>.filled(dim, 0);
    for (int h = 0; h < heads; h++) {
      final scores = List<double>.generate(seqLen, (t) {
        double s = 0;
        for (int j = 0; j < headDim; j++) {
          s += q[h * headDim + j] * kc[t * dim + h * headDim + j];
        }
        return s / math.sqrt(headDim);
      });
      final mx = scores.reduce(math.max);
      final exps = scores.map((s) => math.exp(s - mx)).toList();
      final sum = exps.reduce((a, b) => a + b);
      for (int t = 0; t < seqLen; t++) {
        final p = exps[t] / sum;
        for (int j = 0; j < headDim; j++) {
          attnFlat[h * headDim + j] += p * vc[t * dim + h * headDim + j];
        }
      }
    }
    final attnOut = matVecCpu(woQ.reference, attnFlat, dim, dim);
    final h1 = List<double>.generate(dim, (i) => x[i] + attnOut[i]);

    final xn2 = rmsNormCpu(h1, rms2);
    final gate = matVecCpu(w1Q.reference, xn2, hidden, dim)
        .map((v) => v / (1.0 + math.exp(-v)))
        .toList();
    final up = matVecCpu(w3Q.reference, xn2, hidden, dim);
    final prod = List<double>.generate(hidden, (i) => gate[i] * up[i]);
    final ffn = matVecCpu(w2Q.reference, prod, dim, hidden);
    final expected = List<double>.generate(dim, (i) => h1[i] + ffn[i]);

    // ---- GPU composition --------------------------------------------------
    final wq = await QuantizedTensor.create([dim, dim], GgmlType.q8_0, wqQ.packed);
    final wk = await QuantizedTensor.create([dim, dim], GgmlType.q8_0, wkQ.packed);
    final wv = await QuantizedTensor.create([dim, dim], GgmlType.q8_0, wvQ.packed);
    final wo = await QuantizedTensor.create([dim, dim], GgmlType.q8_0, woQ.packed);
    final w1 = await QuantizedTensor.create([hidden, dim], GgmlType.q8_0, w1Q.packed);
    final w3 = await QuantizedTensor.create([hidden, dim], GgmlType.q8_0, w3Q.packed);
    final w2 = await QuantizedTensor.create([dim, hidden], GgmlType.q8_0, w2Q.packed);

    final rms1T = await Tensor.create([dim], data: rms1);
    final rms2T = await Tensor.create([dim], data: rms2);
    final xT = await Tensor.create([dim], data: xIn);

    final xn1G = await xT.rmsNorm(rms1T, eps: eps);
    final qG = await (await wq.matVec(xn1G))
        .rope(headDim: headDim, heads: heads, positionOffset: pos);
    final kG = await (await wk.matVec(xn1G))
        .rope(headDim: headDim, heads: heads, positionOffset: pos);
    final vG = await wv.matVec(xn1G);

    // KV-cache assembly (CPU concat — GPU append op is a listed gap).
    final kCur = await kG.getData() as Float32List;
    final vCur = await vG.getData() as Float32List;
    final kcData = Float32List(seqLen * dim)
      ..setRange(0, past * dim, pastK)
      ..setRange(past * dim, seqLen * dim, kCur);
    final vcData = Float32List(seqLen * dim)
      ..setRange(0, past * dim, pastV)
      ..setRange(past * dim, seqLen * dim, vCur);
    final kcT = await Tensor.create([seqLen, dim], data: kcData);
    final vcT = await Tensor.create([seqLen, dim], data: vcData);

    // Per-head attention, all heads in one batched pipeline:
    // q: [heads, 1, headDim]; K^T: [heads, headDim, seq]; V: [heads, seq, headDim]
    final q3 = qG.reshape([heads, 1, headDim]);
    final kT3 = await kcT
        .reshape([seqLen, heads, headDim])
        .transpose(axes: [1, 2, 0]); // -> [heads, headDim, seq]
    final v3 = await vcT
        .reshape([seqLen, heads, headDim])
        .transpose(axes: [1, 0, 2]); // -> [heads, seq, headDim]

    final scores = await q3.matMul(kT3); // [heads, 1, seq]
    final scaled = await scores.multiplyScalar(1.0 / math.sqrt(headDim));
    final probs = await scaled.softmax(); // softmax over seq
    final attn3 = await probs.matMul(v3); // [heads, 1, headDim]
    final attnFlatG = attn3.reshape([dim]);

    final attnOutG = await wo.matVec(attnFlatG);
    final h1G = await attnOutG.add(xT); // residual

    final xn2G = await h1G.rmsNorm(rms2T, eps: eps);
    final gateG = await (await w1.matVec(xn2G)).silu();
    final upG = await w3.matVec(xn2G);
    final prodG = await gateG.multiply(upG);
    final ffnG = await w2.matVec(prodG);
    final outG = await h1G.add(ffnG);

    final actual = await outG.getData() as Float32List;
    expect(actual.length, equals(dim));
    for (int i = 0; i < dim; i++) {
      expect(actual[i], closeTo(expected[i], 2e-3),
          reason: 'mismatch at output element $i');
    }
  });
}

// Debug probe: per-stage GPU-vs-CPU comparison of one real expert's FFN to
// localize the moe_test real-file divergence (precision vs indexing bug).
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:gpu_ml/gpu_ml_io.dart';

const q8Path =
    r'C:\models\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf';

Float32List seeded(int n, int seed) {
  final rng = math.Random(seed);
  final out = Float32List(n);
  for (int i = 0; i < n; i++) {
    out[i] = rng.nextDouble() * 2 - 1;
  }
  return out;
}

void stats(String label, Float32List a, List<double> b) {
  double maxAbs = 0, maxMag = 0, rmsNum = 0, rmsDen = 0;
  int maxIdx = 0;
  for (int i = 0; i < a.length; i++) {
    final d = (a[i] - b[i]).abs();
    if (d > maxAbs) {
      maxAbs = d;
      maxIdx = i;
    }
    maxMag = math.max(maxMag, b[i].abs());
    rmsNum += d * d;
    rmsDen += b[i] * b[i];
  }
  print('$label: maxAbsErr=${maxAbs.toStringAsExponential(2)} @$maxIdx '
      'maxMag=${maxMag.toStringAsFixed(3)} '
      'relRms=${math.sqrt(rmsNum / (rmsDen + 1e-30)).toStringAsExponential(2)}');
}

Future<void> main() async {
  final s = await GgufStream.open(q8Path);
  const dim = 2048, ff = 512, expert = 17;

  final gi = s.tensor('blk.0.ffn_gate_exps.weight')!;
  final ui = s.tensor('blk.0.ffn_up_exps.weight')!;
  final di = s.tensor('blk.0.ffn_down_exps.weight')!;

  Future<Float32List> sliceCpu(GgufTensorInfo info, int r, int c) async {
    final traits = ggmlTypeTraits[info.type]!;
    final bytesPer = r * c ~/ traits.blockSize * traits.typeSize;
    final packed = await s.readTensorBytes(info,
        byteOffset: expert * bytesPer, byteLength: bytesPer);
    return dequantizeCpu(info.type, packed, r * c);
  }

  Future<QuantizedTensor> sliceGpu(GgufTensorInfo info, int r, int c) async {
    final traits = ggmlTypeTraits[info.type]!;
    final bytesPer = r * c ~/ traits.blockSize * traits.typeSize;
    final packed = await s.readTensorBytes(info,
        byteOffset: expert * bytesPer, byteLength: bytesPer);
    return QuantizedTensor.create([r, c], info.type, packed);
  }

  final xData = seeded(dim, 40);
  final x = List<double>.generate(dim, (i) => xData[i].toDouble());
  final xT = await Tensor.create([dim], data: xData);

  // CPU (f64) stages.
  final gateW = await sliceCpu(gi, ff, dim);
  final upW = await sliceCpu(ui, ff, dim);
  final downW = await sliceCpu(di, dim, ff);
  List<double> mv(Float32List w, List<double> v, int r, int c) {
    final y = List<double>.filled(r, 0);
    for (int i = 0; i < r; i++) {
      double acc = 0;
      for (int k = 0; k < c; k++) {
        acc += w[i * c + k] * v[k];
      }
      y[i] = acc;
    }
    return y;
  }

  final gCpu = mv(gateW, x, ff, dim);
  final gAct = gCpu.map((v) => v / (1.0 + math.exp(-v))).toList();
  final uCpu = mv(upW, x, ff, dim);
  final prodCpu = List<double>.generate(ff, (i) => gAct[i] * uCpu[i]);
  final outCpu = mv(downW, prodCpu, dim, ff);

  // GPU stages via the SLICED 2D tensors (bypasses expert offset).
  final gateQ2 = await sliceGpu(gi, ff, dim);
  final upQ2 = await sliceGpu(ui, ff, dim);
  final downQ2 = await sliceGpu(di, dim, ff);
  final gGpu = await gateQ2.matVec(xT);
  stats('gate (2D slice)', await gGpu.getData() as Float32List, gCpu);
  final uGpu = await upQ2.matVec(xT);
  stats('up   (2D slice)', await uGpu.getData() as Float32List, uCpu);
  final gA = await gGpu.silu();
  final prodG = await gA.multiply(uGpu);
  stats('prod (2D slice)', await prodG.getData() as Float32List, prodCpu);
  final outG = await downQ2.matVec(prodG);
  stats('down (2D slice)', await outG.getData() as Float32List, outCpu);

  await s.close();
}

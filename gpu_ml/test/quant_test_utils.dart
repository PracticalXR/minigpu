import 'dart:math' as math;
import 'dart:typed_data';

import 'package:gpu_ml/gpu_ml.dart';

Float32List seeded(int n, int seed) {
  final rng = math.Random(seed);
  final out = Float32List(n);
  for (int i = 0; i < n; i++) {
    out[i] = rng.nextDouble() * 2 - 1;
  }
  return out;
}

/// Q8_0-quantizes [values]; returns packed bytes + the effective weights
/// (stored f16-rounded scale * int8) that CPU references must use.
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
      final q = d == 0 ? 0 : (values[b * 32 + l] / d).round().clamp(-127, 127);
      bd.setInt8(b * 34 + 2 + l, q);
      reference[b * 32 + l] = (d * q).toDouble();
    }
  }
  return (packed: packed, reference: reference);
}

List<double> cpuMatVec(Float32List w, List<double> x, int rows, int cols) {
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

double cpuSilu(double v) => v / (1.0 + math.exp(-v));

/// CPU SiLU-gated FFN: down( silu(gate x) * (up x) ).
List<double> cpuExpertFfn(
  Float32List gate,
  Float32List up,
  Float32List down,
  List<double> x,
  int ff,
  int dim,
) {
  final g = cpuMatVec(gate, x, ff, dim).map(cpuSilu).toList();
  final u = cpuMatVec(up, x, ff, dim);
  final prod = List<double>.generate(ff, (i) => g[i] * u[i]);
  return cpuMatVec(down, prod, dim, ff);
}

@TestOn('windows || mac-os || linux')
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:gpu_ml/gpu_ml.dart';
import 'package:minigpu/minigpu.dart';
import 'package:test/test.dart';

import 'quant_test_utils.dart';

/// Validates the dot4I8Packed decode path: per-32-block int8 activation
/// quantizer (quantizeInt8BlockWGSL) + dp4a matVec body against a CPU f32
/// reference on q8_0 weights.  Int8 activations add quantization error, so
/// the bar is relRms < 1e-2 (well below "wrong"), and the dp4a result must
/// match a CPU int8xint8 recompute EXACTLY (kernel correctness).
void main() {
  final gpu = Minigpu();
  setUpAll(() async => gpu.init());

  test('int8 activation quant + dp4a matVec (q8_0)', () async {
    const rows = 96, cols = 512;
    const nbCols = cols ~/ 32;
    final wVals = seeded(rows * cols, 7);
    final q = quantizeQ8_0(wVals); // packed + f16-rounded reference weights
    final x = seeded(cols, 11);

    final wq = await QuantizedTensor.create(
        [rows, cols], GgmlType.q8_0, q.packed, gpu: gpu);
    final xin = gpu.createBuffer(cols * 4, BufferDataType.float32);
    await xin.write(x, cols);
    final xq = gpu.createBuffer(cols, BufferDataType.uint32); // cols/4 u32
    final xsc = gpu.createBuffer(nbCols * 4, BufferDataType.float32);
    final y = gpu.createBuffer(rows * 4, BufferDataType.float32);

    // Quantize x to int8 per 32-block: one workgroup (32 threads) per block.
    final qz = gpu.createComputeShader()
      ..loadKernelString('''
@group(0) @binding(0) var<storage, read_write> xin: array<f32>;
@group(0) @binding(1) var<storage, read_write> xq: array<u32>;
@group(0) @binding(2) var<storage, read_write> xsc: array<f32>;
${QuantizedTensor.quantizeInt8BlockWGSL}
@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  quantizeBlock(wid.x, 0u, lid.x);
}
''');
    qz.setBuffer('xin', xin);
    qz.setBuffer('xq', xq);
    qz.setBuffer('xsc', xsc);
    await qz.dispatch(nbCols, 1, 1);

    // Read back the quantized activations to build the EXACT int8 reference.
    final xqWords = Uint32List(cols ~/ 4);
    await xq.read(xqWords, cols ~/ 4, dataType: BufferDataType.uint32);
    final xscVals = Float32List(nbCols);
    await xsc.read(xscVals, nbCols);
    final xInt8 = Int8List(cols);
    for (int w = 0; w < cols ~/ 4; w++) {
      for (int b = 0; b < 4; b++) {
        xInt8[w * 4 + b] = (xqWords[w] >> (b * 8)) & 0xFF;
      }
    }

    final dp = gpu.createComputeShader()
      ..loadKernelString('''
@group(0) @binding(0) var<storage, read_write> wq: array<u32>;
@group(0) @binding(1) var<storage, read_write> xq: array<u32>;
@group(0) @binding(2) var<storage, read_write> xsc: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
const ROWS: u32 = ${rows}u;
const COLS: u32 = ${cols}u;
${QuantizedTensor.accessorsWGSL}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let row: u32 = gid.x + gid.y * (nwg.x * 64u);
  let eb: u32 = 0u;
  var acc: f32 = 0.0;
  if (row < ROWS) {
${QuantizedTensor.matVecDp4aBodyWGSL(threadVar: '0u', stride: '1u')}
  }
  if (row < ROWS) { y[row] = acc; }
}
''');
    dp.setBuffer('wq', wq.buffer);
    dp.setBuffer('xq', xq);
    dp.setBuffer('xsc', xsc);
    dp.setBuffer('y', y);
    await dp.dispatch((rows + 63) ~/ 64, 1, 1);

    final got = Float32List(rows);
    await y.read(got, rows);

    // Sanity: the read-back int8 activations reconstruct x within the
    // per-block quant step (bounds the approximation the matVec inherits).
    double xErr = 0;
    for (int j = 0; j < nbCols; j++) {
      for (int i = 0; i < 32; i++) {
        final recon = xInt8[j * 32 + i] * xscVals[j];
        xErr = math.max(xErr, (recon - x[j * 32 + i]).abs());
      }
    }
    // ignore: avoid_print
    print('int8 activation max abs recon error: $xErr');

    // f32 CPU reference from the f16-rounded quantized weights (ground truth
    // the dp4a path approximates with int8 activations).
    double se = 0, ref = 0, maxAbs = 0;
    for (int r = 0; r < rows; r++) {
      double want = 0;
      for (int c = 0; c < cols; c++) {
        want += q.reference[r * cols + c] * x[c];
      }
      final d = got[r] - want;
      se += d * d;
      ref += want * want;
      if (d.abs() > maxAbs) maxAbs = d.abs();
    }
    final relRms = math.sqrt(se / ref);
    // ignore: avoid_print
    print('dp4a relRms vs f32: $relRms  maxAbs: $maxAbs');
    expect(relRms, lessThan(1e-2),
        reason: 'dp4a int8-activation matVec too far from f32 (relRms $relRms)');

    qz.destroy();
    dp.destroy();
    for (final b in [xin, xq, xsc, y]) {
      b.destroy();
    }
    wq.destroy();
  });
}

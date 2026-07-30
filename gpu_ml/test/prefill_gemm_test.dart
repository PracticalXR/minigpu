@TestOn('windows || mac-os || linux')
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:gpu_ml/gpu_ml.dart';
import 'package:minigpu/minigpu.dart';
import 'package:test/test.dart';

import 'quant_test_utils.dart';

/// Standalone validation of the prefill dense-GEMM path's two kernels
/// (dequant-to-packed-f16 + 16x16 tiled GEMM), byte-for-byte the same WGSL
/// the plan builds, against a CPU reference.  Isolates kernel math from the
/// plan wiring.
void main() {
  final gpu = Minigpu();

  setUpAll(() async {
    await gpu.init();
  });

  test('dequantF16 + tiled GEMM match CPU (Q8_0)', () async {
    const rows = 48, cols = 64, T = 20;
    final wVals = seeded(rows * cols, 7);
    final q = quantizeQ8_0(wVals);
    final x = seeded(T * cols, 11);

    final wq = await QuantizedTensor.create(
        [rows, cols], GgmlType.q8_0, q.packed, gpu: gpu);
    final w16 = gpu.createBuffer(rows * cols * 2, BufferDataType.uint32);
    final xBuf = gpu.createBuffer(T * cols * 4, BufferDataType.float32);
    await xBuf.write(x, T * cols);
    final yBuf = gpu.createBuffer(T * rows * 4, BufferDataType.float32);
    final pT = gpu.createBuffer(16, BufferDataType.uint32);
    await pT.write(Uint32List.fromList([T, 0, 0, 0]), 4,
        dataType: BufferDataType.uint32);

    const n = rows * cols;
    final deq = gpu.createComputeShader()
      ..loadKernelString('''
@group(0) @binding(0) var<storage, read_write> wq: array<u32>;
@group(0) @binding(1) var<storage, read_write> w16: array<u32>;

${QuantizedTensor.accessorsWGSL}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let blk: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (blk >= ${n ~/ 32}u) { return; }
  let base: u32 = blk * 34u;
  let d: f32 = f16At(base);
  let qb: u32 = base + 2u;
  let qw: u32 = qb >> 2u;
  let qs: u32 = (qb & 3u) * 8u;
  var carry: u32 = wq[qw];
  for (var k: u32 = 0u; k < 8u; k = k + 1u) {
    let nxt: u32 = wq[qw + k + 1u];
    var raw: u32;
    if (qs == 0u) { raw = carry; } else { raw = (carry >> qs) | (nxt << (32u - qs)); }
    carry = nxt;
    let v0: f32 = d * f32(bitcast<i32>(raw << 24u) >> 24u);
    let v1: f32 = d * f32(bitcast<i32>(raw << 16u) >> 24u);
    let v2: f32 = d * f32(bitcast<i32>(raw << 8u) >> 24u);
    let v3: f32 = d * f32(bitcast<i32>(raw) >> 24u);
    let ow: u32 = blk * 16u + k * 2u;
    w16[ow] = pack2x16float(vec2<f32>(v0, v1));
    w16[ow + 1u] = pack2x16float(vec2<f32>(v2, v3));
  }
}
''');
    deq.setBuffer('wq', wq.buffer);
    deq.setBuffer('w16', w16);
    await deq.dispatch((n ~/ 32 + 255) ~/ 256, 1, 1);

    // Validate the dequant stage on its own first.
    final packedWords = Uint32List(n ~/ 2);
    await w16.read(packedWords, n ~/ 2, dataType: BufferDataType.uint32);
    for (int e = 0; e < 8; e++) {
      final word = packedWords[e >> 1];
      final bits = (e & 1) == 1 ? (word >> 16) & 0xFFFF : word & 0xFFFF;
      // ignore: avoid_print
      print('e=$e got=${halfBitsToFloat(bits)} want=${q.reference[e]}');
    }
    for (int e = 64; e < 68; e++) {
      final word = packedWords[e >> 1];
      final bits = (e & 1) == 1 ? (word >> 16) & 0xFFFF : word & 0xFFFF;
      // ignore: avoid_print
      print('e=$e got=${halfBitsToFloat(bits)} want=${q.reference[e]}');
    }
    double deqMax = 0;
    for (int e = 0; e < n; e++) {
      final word = packedWords[e >> 1];
      final bits = (e & 1) == 1 ? (word >> 16) & 0xFFFF : word & 0xFFFF;
      final v = halfBitsToFloat(bits);
      final want = q.reference[e];
      final rel = (v - want).abs() / math.max(want.abs(), 1e-3);
      if (rel > deqMax) deqMax = rel;
    }
    expect(deqMax, lessThan(2e-2),
        reason: 'dequant stage diverges (maxRel $deqMax)');

    final gemm = gpu.createComputeShader()
      ..loadKernelString('''
@group(0) @binding(0) var<storage, read_write> w16: array<u32>;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> pT: array<u32>;

const R: u32 = ${rows}u;
const C: u32 = ${cols}u;

var<workgroup> Xs: array<f32, 256>;
var<workgroup> Ws: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let T: u32 = pT[0];
  let row: u32 = wid.x * 16u + lid.x;
  let tok: u32 = wid.y * 16u + lid.y;
  var acc: f32 = 0.0;
  for (var c0: u32 = 0u; c0 < C; c0 = c0 + 16u) {
    let xt: u32 = wid.y * 16u + lid.y;
    var xv: f32 = 0.0;
    if (xt < T) { xv = x[xt * C + c0 + lid.x]; }
    Xs[lid.y * 16u + lid.x] = xv;
    let wr: u32 = wid.x * 16u + lid.y;
    var wv: f32 = 0.0;
    if (wr < R) {
      let e: u32 = wr * C + c0 + lid.x;
      let pair: vec2<f32> = unpack2x16float(w16[e >> 1u]);
      wv = select(pair.x, pair.y, (e & 1u) == 1u);
    }
    Ws[lid.y * 16u + lid.x] = wv;
    workgroupBarrier();
    for (var k: u32 = 0u; k < 16u; k = k + 1u) {
      acc = acc + Xs[lid.y * 16u + k] * Ws[lid.x * 16u + k];
    }
    workgroupBarrier();
  }
  if (row < R && tok < T) { y[tok * R + row] = acc; }
}
''');
    gemm.setBuffer('w16', w16);
    gemm.setBuffer('x', xBuf);
    gemm.setBuffer('y', yBuf);
    gemm.setBuffer('pT', pT);
    await gemm.dispatch((rows + 15) ~/ 16, (T + 15) ~/ 16, 1);

    final got = Float32List(T * rows);
    await yBuf.read(got, T * rows);

    // The GPU path stores weights as f16 — round the reference identically.
    final ref16 = Float32List(rows * cols);
    for (int e = 0; e < rows * cols; e++) {
      ref16[e] = halfBitsToFloat(floatToHalfBits(q.reference[e]));
    }
    double se = 0, ref = 0;
    for (int t = 0; t < T; t++) {
      final xt = [for (int c = 0; c < cols; c++) x[t * cols + c].toDouble()];
      final want = cpuMatVec(ref16, xt, rows, cols);
      for (int r = 0; r < rows; r++) {
        final d = got[t * rows + r] - want[r];
        se += d * d;
        ref += want[r] * want[r];
      }
    }
    final relRms = math.sqrt(se / ref);
    // ignore: avoid_print
    print('gemm relRms: $relRms');
    // f16-rounded weights: tolerance well above f32 noise, well below wrong.
    expect(relRms, lessThan(2e-3),
        reason: 'dequant+GEMM diverges from CPU (relRms $relRms)');

    deq.destroy();
    gemm.destroy();
    w16.destroy();
    xBuf.destroy();
    yBuf.destroy();
    pT.destroy();
    wq.destroy();
  });
}

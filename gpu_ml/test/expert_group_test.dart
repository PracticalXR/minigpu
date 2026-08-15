@TestOn('windows || mac-os || linux')
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:gpu_ml/gpu_ml.dart';
import 'package:minigpu/minigpu.dart';
import 'package:test/test.dart';

import 'quant_test_utils.dart';

/// Standalone validation of the prefill expert-grouping kernels (count/scan/
/// worklist prep, scatter, grouped tile matvec), byte-for-byte the same WGSL
/// the plan builds, against a CPU reference.  Pins the multi-token funnel-
/// load pattern against FXC miscompiles (see the lane-3 byteAt history).
void main() {
  final gpu = Minigpu();

  setUpAll(() async {
    await gpu.init();
  });

  test('grouped expert matvec matches CPU (Q8_0)', () async {
    const E = 8, rows = 48, cols = 512, T = 20, K = 4;
    const TB = 16, TPR = 4, RPW = 64;
    const TK = T * K;
    const bpe = rows * cols ~/ 32 * 34;

    // Expert stack + per-token distinct top-K assignments.
    final rng = math.Random(3);
    final stackBytes = Uint8List(E * bpe);
    final refW = List<Float32List>.generate(E, (e) {
      final q = quantizeQ8_0(seeded(rows * cols, 100 + e));
      stackBytes.setRange(e * bpe, (e + 1) * bpe, q.packed);
      return q.reference;
    });
    final idx = Uint32List(TK);
    for (int t = 0; t < T; t++) {
      final picks = List<int>.generate(E, (i) => i)..shuffle(rng);
      for (int s = 0; s < K; s++) {
        idx[t * K + s] = picks[s];
      }
    }
    final x = seeded(T * cols, 55);

    final wq = gpu.createBuffer(stackBytes.length, BufferDataType.uint32);
    await wq.write(
        Uint32List.sublistView(stackBytes), stackBytes.length ~/ 4,
        dataType: BufferDataType.uint32);
    final idxb = gpu.createBuffer(TK * 4, BufferDataType.uint32);
    await idxb.write(idx, TK, dataType: BufferDataType.uint32);
    final pT = gpu.createBuffer(16, BufferDataType.uint32);
    await pT.write(Uint32List.fromList([T, 0, 0, 0]), 4,
        dataType: BufferDataType.uint32);
    final off = gpu.createBuffer((E + 1) * 4, BufferDataType.uint32);
    const wlCap = TK ~/ TB + E + 1;
    final wl = gpu.createBuffer(wlCap * 4, BufferDataType.uint32);
    final wlc = gpu.createBuffer(16, BufferDataType.uint32);
    final srt = gpu.createBuffer(TK * 4, BufferDataType.uint32);
    final xBuf = gpu.createBuffer(T * cols * 4, BufferDataType.float32);
    await xBuf.write(x, T * cols);
    final yBuf = gpu.createBuffer(TK * rows * 4, BufferDataType.float32);

    final prep = gpu.createComputeShader()
      ..loadKernelString('''
@group(0) @binding(0) var<storage, read_write> idxb: array<u32>;
@group(0) @binding(1) var<storage, read_write> pT: array<u32>;
@group(0) @binding(2) var<storage, read_write> off: array<u32>;
@group(0) @binding(3) var<storage, read_write> wl: array<u32>;
@group(0) @binding(4) var<storage, read_write> wlc: array<u32>;
@group(0) @binding(5) var<storage, read_write> srt: array<u32>;

const E: u32 = ${E}u;
const K: u32 = ${K}u;
const TB: u32 = ${TB}u;

var<workgroup> scnt: array<u32, $E>;
var<workgroup> soff: array<u32, $E>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tk: u32 = pT[0] * K;
  let e: u32 = lid.x;
  if (e < E) {
    var c: u32 = 0u;
    for (var z: u32 = 0u; z < tk; z = z + 1u) {
      if (idxb[z] == e) { c = c + 1u; }
    }
    scnt[e] = c;
  }
  workgroupBarrier();
  if (lid.x == 0u) {
    var run: u32 = 0u;
    var w: u32 = 0u;
    for (var ee: u32 = 0u; ee < E; ee = ee + 1u) {
      off[ee] = run;
      soff[ee] = run;
      let c: u32 = scnt[ee];
      run = run + c;
      for (var s: u32 = 0u; s < c; s = s + TB) {
        wl[w] = (ee << 16u) | s;
        w = w + 1u;
      }
    }
    off[E] = run;
    wlc[0] = w;
  }
  workgroupBarrier();
  if (e < E) {
    var p: u32 = soff[e];
    for (var z: u32 = 0u; z < tk; z = z + 1u) {
      if (idxb[z] == e) {
        srt[p] = z;
        p = p + 1u;
      }
    }
  }
}
''');
    prep.setBuffer('idxb', idxb);
    prep.setBuffer('pT', pT);
    prep.setBuffer('off', off);
    prep.setBuffer('wl', wl);
    prep.setBuffer('wlc', wlc);
    prep.setBuffer('srt', srt);
    await prep.dispatch(1, 1, 1);

    // Sanity-check the grouping itself before the matvec.
    final offV = Uint32List(E + 1);
    await off.read(offV, E + 1, dataType: BufferDataType.uint32);
    final srtV = Uint32List(TK);
    await srt.read(srtV, TK, dataType: BufferDataType.uint32);
    expect(offV[E], TK);
    final seen = <int>{};
    for (int e = 0; e < E; e++) {
      for (int p = offV[e]; p < offV[e + 1]; p++) {
        expect(idx[srtV[p]], e, reason: 'sorted[$p] not in expert $e group');
        expect(seen.add(srtV[p]), isTrue, reason: 'duplicate z ${srtV[p]}');
      }
    }
    expect(seen.length, TK);

    // Unrolled per-token fragments, generated exactly as the plan does —
    // dynamically-indexed local arrays land in FXC indexable temps (slow).
    final jts = List.generate(TB, (i) => i);
    final accDecl = jts.map((i) => '  var acc$i: f32 = 0.0;').join('\n');
    final bsumDecl =
        jts.map((i) => '      var bsum$i: f32 = 0.0;').join('\n');
    final macs = jts
        .map((i) =>
            '          bsum$i = bsum$i + dot(qv, xs[xsb + ${i * 8}u + k]);')
        .join('\n');
    final accAdd =
        jts.map((i) => '        acc$i = acc$i + d * bsum$i;').join('\n');
    final redStore = jts
        .map((i) => '  red[(lid.y * TB + ${i}u) * TPR + lid.x] = acc$i;')
        .join('\n');

    final grouped = gpu.createComputeShader()
      ..loadKernelString('''
@group(0) @binding(0) var<storage, read_write> wq: array<u32>;
@group(0) @binding(1) var<storage, read_write> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> srt: array<u32>;
@group(0) @binding(4) var<storage, read_write> off: array<u32>;
@group(0) @binding(5) var<storage, read_write> wl: array<u32>;
@group(0) @binding(6) var<storage, read_write> wlc: array<u32>;

const ROWS: u32 = ${rows}u;
const COLS: u32 = ${cols}u;
const K: u32 = ${K}u;
const TB: u32 = ${TB}u;
const TPR: u32 = ${TPR}u;
const RPW: u32 = ${RPW}u;

${QuantizedTensor.accessorsWGSL}

var<workgroup> zs: array<u32, $TB>;
var<workgroup> xb: array<u32, $TB>;
var<workgroup> xs: array<vec4<f32>, ${TPR * 8 * TB}>;
var<workgroup> red: array<f32, ${TB * TPR * RPW}>;

@compute @workgroup_size($TPR, $RPW)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  var n: u32 = 0u;
  var e: u32 = 0u;
  var p0: u32 = 0u;
  if (wid.y < wlc[0]) {
    let entry: u32 = wl[wid.y];
    e = entry >> 16u;
    let ls: u32 = entry & 0xFFFFu;
    let c: u32 = off[e + 1u] - off[e];
    n = min(TB, c - ls);
    p0 = off[e] + ls;
  }
  let tid: u32 = lid.y * TPR + lid.x;
  if (tid < TB) {
    var p: u32 = p0;
    if (tid < n) { p = p0 + tid; }
    var z: u32 = 0u;
    if (n > 0u) { z = srt[p]; }
    zs[tid] = z;
    xb[tid] = (z / K) * COLS / 4u;
  }
  workgroupBarrier();
  let row: u32 = wid.x * RPW + lid.y;
  let eb: u32 = e * ${bpe}u;
  let nb: u32 = COLS / 32u;
  let xsb: u32 = lid.x * ${TB * 8}u;
$accDecl
  for (var r: u32 = 0u; r < ${(cols ~/ 32 + TPR - 1) ~/ TPR}u; r = r + 1u) {
    for (var si: u32 = tid; si < ${TPR * 8 * TB}u; si = si + ${TPR * RPW}u) {
      let jj: u32 = r * TPR + si / ${TB * 8}u;
      if (jj < nb) {
        xs[si] = x[xb[(si / 8u) % TB] + jj * 8u + (si % 8u)];
      }
    }
    workgroupBarrier();
    let j: u32 = r * TPR + lid.x;
    if (row < ROWS && n > 0u && j < nb) {
      let base: u32 = eb + (row * nb + j) * 34u;
      let d: f32 = f16At(base);
      let qb: u32 = base + 2u;
      let qw: u32 = qb >> 2u;
      let qs: u32 = (qb & 3u) * 8u;
      var carry: u32 = wq[qw];
$bsumDecl
      for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let nxt: u32 = wq[qw + k + 1u];
        var raw: u32;
        if (qs == 0u) { raw = carry; } else { raw = (carry >> qs) | (nxt << (32u - qs)); }
        carry = nxt;
        let qv: vec4<f32> = vec4<f32>(
            f32(bitcast<i32>(raw << 24u) >> 24u),
            f32(bitcast<i32>(raw << 16u) >> 24u),
            f32(bitcast<i32>(raw << 8u) >> 24u),
            f32(bitcast<i32>(raw) >> 24u));
$macs
      }
$accAdd
    }
    workgroupBarrier();
  }
$redStore
  workgroupBarrier();
  for (var pp: u32 = tid; pp < RPW * TB; pp = pp + ${TPR * RPW}u) {
    let rw: u32 = pp / TB;
    let jt: u32 = pp % TB;
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < TPR; i = i + 1u) {
      sum = sum + red[(rw * TB + jt) * TPR + i];
    }
    let orow: u32 = wid.x * RPW + rw;
    if (orow < ROWS && jt < n) {
      y[zs[jt] * ROWS + orow] = sum;
    }
  }
}
''');
    grouped.setBuffer('wq', wq);
    grouped.setBuffer('x', xBuf);
    grouped.setBuffer('y', yBuf);
    grouped.setBuffer('srt', srt);
    grouped.setBuffer('off', off);
    grouped.setBuffer('wl', wl);
    grouped.setBuffer('wlc', wlc);
    const rowGroups = (rows + RPW - 1) ~/ RPW;
    const wlBound = (TK + TB - 1) ~/ TB + E;
    await grouped.dispatch(rowGroups, wlBound, 1);

    final got = Float32List(TK * rows);
    await yBuf.read(got, TK * rows);

    double se = 0, ref = 0, maxAbs = 0;
    for (int z = 0; z < TK; z++) {
      final t = z ~/ K, e = idx[z];
      final xt = [for (int c = 0; c < cols; c++) x[t * cols + c].toDouble()];
      final want = cpuMatVec(refW[e], xt, rows, cols);
      for (int r = 0; r < rows; r++) {
        final d = got[z * rows + r] - want[r];
        se += d * d;
        ref += want[r] * want[r];
        if (d.abs() > maxAbs) maxAbs = d.abs();
      }
    }
    final relRms = math.sqrt(se / ref);
    // ignore: avoid_print
    print('grouped relRms: $relRms  maxAbs: $maxAbs');
    // Exact same quantized weights as the reference: only f32 sum-order noise.
    expect(relRms, lessThan(1e-5),
        reason: 'grouped expert matvec diverges (relRms $relRms)');

    prep.destroy();
    grouped.destroy();
    for (final b in [wq, idxb, pT, off, wl, wlc, srt, xBuf, yBuf]) {
      b.destroy();
    }
  });

  test('grouped dp4a expert matvec matches integer CPU reference', () async {
    const E = 8, rows = 48, cols = 512, T = 20, K = 4;
    const TB = 16, TPR = 4, RPW = 64;
    const TK = T * K;
    const bpe = rows * cols ~/ 32 * 34;
    const nbRow = cols ~/ 32; // 16 blocks per activation row
    const nbTotal = T * nbRow; // 320 activation blocks

    final rng = math.Random(7);
    final stackBytes = Uint8List(E * bpe);
    for (int e = 0; e < E; e++) {
      final q = quantizeQ8_0(seeded(rows * cols, 300 + e));
      stackBytes.setRange(e * bpe, (e + 1) * bpe, q.packed);
    }
    final idx = Uint32List(TK);
    for (int t = 0; t < T; t++) {
      final picks = List<int>.generate(E, (i) => i)..shuffle(rng);
      for (int s = 0; s < K; s++) {
        idx[t * K + s] = picks[s];
      }
    }
    final x = seeded(T * cols, 77);

    // CPU int8 activation quantization — same math as plan-quantxb.
    final xqInts = Int32List(T * cols);
    final xscRef = Float32List(nbTotal);
    for (int b = 0; b < nbTotal; b++) {
      double mx = 0;
      for (int i = 0; i < 32; i++) {
        mx = math.max(mx, x[b * 32 + i].abs());
      }
      final scale = mx / 127.0;
      xscRef[b] = scale;
      final inv = mx > 0 ? 1.0 / scale : 0.0;
      for (int i = 0; i < 32; i++) {
        xqInts[b * 32 + i] =
            (x[b * 32 + i] * inv).roundToDouble().clamp(-127, 127).toInt();
      }
    }
    // CPU integer-dot reference (f64 accumulate; GPU differs only by f32
    // outer sum order).
    final bd = ByteData.sublistView(stackBytes);
    final want = Float64List(TK * rows);
    for (int z = 0; z < TK; z++) {
      final t = z ~/ K, e = idx[z];
      for (int r = 0; r < rows; r++) {
        double acc = 0;
        for (int j = 0; j < nbRow; j++) {
          final base = e * bpe + (r * nbRow + j) * 34;
          final d = halfBitsToFloat(bd.getUint16(base, Endian.little));
          var isum = 0;
          for (int i = 0; i < 32; i++) {
            isum += bd.getInt8(base + 2 + i) * xqInts[(t * nbRow + j) * 32 + i];
          }
          acc += d * xscRef[t * nbRow + j] * isum;
        }
        want[z * rows + r] = acc;
      }
    }

    final wq = gpu.createBuffer(stackBytes.length, BufferDataType.uint32);
    await wq.write(Uint32List.sublistView(stackBytes), stackBytes.length ~/ 4,
        dataType: BufferDataType.uint32);
    final idxb = gpu.createBuffer(TK * 4, BufferDataType.uint32);
    await idxb.write(idx, TK, dataType: BufferDataType.uint32);
    final pT = gpu.createBuffer(16, BufferDataType.uint32);
    await pT.write(Uint32List.fromList([T, 0, 0, 0]), 4,
        dataType: BufferDataType.uint32);
    final off = gpu.createBuffer((E + 1) * 4, BufferDataType.uint32);
    const wlCap = TK ~/ TB + E + 1;
    final wl = gpu.createBuffer(wlCap * 4, BufferDataType.uint32);
    final wlc = gpu.createBuffer(16, BufferDataType.uint32);
    final srt = gpu.createBuffer(TK * 4, BufferDataType.uint32);
    final xBuf = gpu.createBuffer(T * cols * 4, BufferDataType.float32);
    await xBuf.write(x, T * cols);
    final xqBuf = gpu.createBuffer(T * cols, BufferDataType.uint32);
    final xscBuf = gpu.createBuffer(nbTotal * 4, BufferDataType.float32);
    final yBuf = gpu.createBuffer(TK * rows * 4, BufferDataType.float32);

    // Prep (same as test 1 — count/scan/worklist/scatter, no atomics).
    final prep = gpu.createComputeShader()
      ..loadKernelString('''
@group(0) @binding(0) var<storage, read_write> idxb: array<u32>;
@group(0) @binding(1) var<storage, read_write> pT: array<u32>;
@group(0) @binding(2) var<storage, read_write> off: array<u32>;
@group(0) @binding(3) var<storage, read_write> wl: array<u32>;
@group(0) @binding(4) var<storage, read_write> wlc: array<u32>;
@group(0) @binding(5) var<storage, read_write> srt: array<u32>;

const E: u32 = ${E}u;
const K: u32 = ${K}u;
const TB: u32 = ${TB}u;

var<workgroup> scnt: array<u32, $E>;
var<workgroup> soff: array<u32, $E>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tk: u32 = pT[0] * K;
  let e: u32 = lid.x;
  if (e < E) {
    var c: u32 = 0u;
    for (var z: u32 = 0u; z < tk; z = z + 1u) {
      if (idxb[z] == e) { c = c + 1u; }
    }
    scnt[e] = c;
  }
  workgroupBarrier();
  if (lid.x == 0u) {
    var run: u32 = 0u;
    var w: u32 = 0u;
    for (var ee: u32 = 0u; ee < E; ee = ee + 1u) {
      off[ee] = run;
      soff[ee] = run;
      let c: u32 = scnt[ee];
      run = run + c;
      for (var s: u32 = 0u; s < c; s = s + TB) {
        wl[w] = (ee << 16u) | s;
        w = w + 1u;
      }
    }
    off[E] = run;
    wlc[0] = w;
  }
  workgroupBarrier();
  if (e < E) {
    var p: u32 = soff[e];
    for (var z: u32 = 0u; z < tk; z = z + 1u) {
      if (idxb[z] == e) {
        srt[p] = z;
        p = p + 1u;
      }
    }
  }
}
''');
    prep.setBuffer('idxb', idxb);
    prep.setBuffer('pT', pT);
    prep.setBuffer('off', off);
    prep.setBuffer('wl', wl);
    prep.setBuffer('wlc', wlc);
    prep.setBuffer('srt', srt);
    await prep.dispatch(1, 1, 1);

    // Bulk quantizer — byte-for-byte plan-quantxb.
    final quant = gpu.createComputeShader()
      ..loadKernelString('''
@group(0) @binding(0) var<storage, read_write> xin: array<f32>;
@group(0) @binding(1) var<storage, read_write> xq: array<u32>;
@group(0) @binding(2) var<storage, read_write> xsc: array<f32>;

var<workgroup> amax: array<f32, 256>;
var<workgroup> qsh: array<i32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let cl: u32 = lid.x / 32u;
  let ln: u32 = lid.x % 32u;
  let blk: u32 = (wid.x + wid.y * nwg.x) * 8u + cl;
  let ok: bool = blk < ${nbTotal}u;
  var v: f32 = 0.0;
  if (ok) { v = xin[blk * 32u + ln]; }
  amax[lid.x] = abs(v);
  workgroupBarrier();
  for (var s: u32 = 16u; s > 0u; s = s >> 1u) {
    if (ln < s) { amax[lid.x] = max(amax[lid.x], amax[lid.x + s]); }
    workgroupBarrier();
  }
  let mx: f32 = amax[cl * 32u];
  let scale: f32 = mx / 127.0;
  let inv: f32 = select(0.0, 1.0 / scale, mx > 0.0);
  qsh[lid.x] = clamp(i32(round(v * inv)), -127, 127);
  workgroupBarrier();
  if (ln < 8u && ok) {
    let b: u32 = cl * 32u + ln * 4u;
    xq[blk * 8u + ln] = (u32(qsh[b]) & 0xFFu) | ((u32(qsh[b + 1u]) & 0xFFu) << 8u)
        | ((u32(qsh[b + 2u]) & 0xFFu) << 16u) | ((u32(qsh[b + 3u]) & 0xFFu) << 24u);
  }
  if (ln == 0u && ok) { xsc[blk] = scale; }
}
''');
    quant.setBuffer('xin', xBuf);
    quant.setBuffer('xq', xqBuf);
    quant.setBuffer('xsc', xscBuf);
    await quant.dispatch((nbTotal + 7) ~/ 8, 1, 1);

    // Grouped dp4a matvec — generated exactly as the plan does (gate-style,
    // xPerZ=false).
    final jts = List.generate(TB, (i) => i);
    final accDecl = jts.map((i) => '  var acc$i: f32 = 0.0;').join('\n');
    final isumDecl = jts.map((i) => '      var isum$i: i32 = 0;').join('\n');
    final macs = jts
        .map((i) =>
            '        isum$i = isum$i + dot4I8Packed(raw, xsq[xqb + ${i * 8}u + k]);')
        .join('\n');
    final accAdd = jts
        .map((i) =>
            '        acc$i = acc$i + d * xsc[ssb[${i}u] + j] * f32(isum$i);')
        .join('\n');
    final redStore = jts
        .map((i) => '  red[(lid.y * TB + ${i}u) * TPR + lid.x] = acc$i;')
        .join('\n');
    final grouped = gpu.createComputeShader()
      ..loadKernelString('''
@group(0) @binding(0) var<storage, read_write> wq: array<u32>;
@group(0) @binding(1) var<storage, read_write> xq: array<u32>;
@group(0) @binding(2) var<storage, read_write> xsc: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read_write> srt: array<u32>;
@group(0) @binding(5) var<storage, read_write> off: array<u32>;
@group(0) @binding(6) var<storage, read_write> wl: array<u32>;
@group(0) @binding(7) var<storage, read_write> wlc: array<u32>;

const ROWS: u32 = ${rows}u;
const COLS: u32 = ${cols}u;
const K: u32 = ${K}u;
const TB: u32 = ${TB}u;
const TPR: u32 = ${TPR}u;
const RPW: u32 = ${RPW}u;

${QuantizedTensor.accessorsWGSL}

var<workgroup> zs: array<u32, $TB>;
var<workgroup> swb: array<u32, $TB>;
var<workgroup> ssb: array<u32, $TB>;
var<workgroup> xsq: array<u32, ${TPR * 8 * TB}>;
var<workgroup> red: array<f32, ${TB * TPR * RPW}>;

@compute @workgroup_size($TPR, $RPW)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  var n: u32 = 0u;
  var e: u32 = 0u;
  var p0: u32 = 0u;
  if (wid.y < wlc[0]) {
    let entry: u32 = wl[wid.y];
    e = entry >> 16u;
    let ls: u32 = entry & 0xFFFFu;
    let c: u32 = off[e + 1u] - off[e];
    n = min(TB, c - ls);
    p0 = off[e] + ls;
  }
  let tid: u32 = lid.y * TPR + lid.x;
  if (tid < TB) {
    var p: u32 = p0;
    if (tid < n) { p = p0 + tid; }
    var z: u32 = 0u;
    if (n > 0u) { z = srt[p]; }
    zs[tid] = z;
    let xrow: u32 = (z / K);
    swb[tid] = xrow * (COLS / 4u);
    ssb[tid] = xrow * (COLS / 32u);
  }
  workgroupBarrier();
  let row: u32 = wid.x * RPW + lid.y;
  let eb: u32 = e * ${bpe}u;
  let nb: u32 = COLS / 32u;
  let xqb: u32 = lid.x * ${TB * 8}u;
$accDecl
  for (var r: u32 = 0u; r < ${(cols ~/ 32 + TPR - 1) ~/ TPR}u; r = r + 1u) {
    for (var si: u32 = tid; si < ${TPR * 8 * TB}u; si = si + ${TPR * RPW}u) {
      let jj: u32 = r * TPR + si / ${TB * 8}u;
      if (jj < nb) {
        xsq[si] = xq[swb[(si / 8u) % TB] + jj * 8u + (si % 8u)];
      }
    }
    workgroupBarrier();
    let j: u32 = r * TPR + lid.x;
    if (row < ROWS && n > 0u && j < nb) {
      let base: u32 = eb + (row * nb + j) * 34u;
      let d: f32 = f16At(base);
      let qb: u32 = base + 2u;
      let qw: u32 = qb >> 2u;
      let qs: u32 = (qb & 3u) * 8u;
      var carry: u32 = wq[qw];
$isumDecl
      for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let nxt: u32 = wq[qw + k + 1u];
        var raw: u32;
        if (qs == 0u) { raw = carry; } else { raw = (carry >> qs) | (nxt << (32u - qs)); }
        carry = nxt;
$macs
      }
$accAdd
    }
    workgroupBarrier();
  }
$redStore
  workgroupBarrier();
  for (var pp: u32 = tid; pp < RPW * TB; pp = pp + ${TPR * RPW}u) {
    let rw: u32 = pp / TB;
    let jt: u32 = pp % TB;
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < TPR; i = i + 1u) {
      sum = sum + red[(rw * TB + jt) * TPR + i];
    }
    let orow: u32 = wid.x * RPW + rw;
    if (orow < ROWS && jt < n) {
      y[zs[jt] * ROWS + orow] = sum;
    }
  }
}
''');
    grouped.setBuffer('wq', wq);
    grouped.setBuffer('xq', xqBuf);
    grouped.setBuffer('xsc', xscBuf);
    grouped.setBuffer('y', yBuf);
    grouped.setBuffer('srt', srt);
    grouped.setBuffer('off', off);
    grouped.setBuffer('wl', wl);
    grouped.setBuffer('wlc', wlc);
    const rowGroups = (rows + RPW - 1) ~/ RPW;
    const wlBound = (TK + TB - 1) ~/ TB + E;
    await grouped.dispatch(rowGroups, wlBound, 1);

    final got = Float32List(TK * rows);
    await yBuf.read(got, TK * rows);

    double se = 0, ref = 0, maxAbs = 0;
    for (int i = 0; i < TK * rows; i++) {
      final d = got[i] - want[i];
      se += d * d;
      ref += want[i] * want[i];
      if (d.abs() > maxAbs) maxAbs = d.abs();
    }
    final relRms = math.sqrt(se / ref);
    // ignore: avoid_print
    print('grouped-dp4a relRms: $relRms  maxAbs: $maxAbs');
    // Integer dots are exact; only f32 outer-sum order differs from the f64
    // CPU reference.
    expect(relRms, lessThan(1e-5),
        reason: 'grouped dp4a matvec diverges (relRms $relRms)');

    prep.destroy();
    quant.destroy();
    grouped.destroy();
    for (final b in [wq, idxb, pT, off, wl, wlc, srt, xBuf, xqBuf, xscBuf, yBuf]) {
      b.destroy();
    }
  });

  test('dense int8 GEMM matches integer CPU reference', () async {
    const rows = 48, cols = 512, T = 20;
    const TB = 16, TPR = 4, RPW = 64;
    const nbRow = cols ~/ 32;
    const nbTotal = T * nbRow;

    final qw8 = quantizeQ8_0(seeded(rows * cols, 900));
    final x = seeded(T * cols, 901);

    // CPU int8 activation quantization (same math as plan-quantxb).
    final xqInts = Int32List(T * cols);
    final xscRef = Float32List(nbTotal);
    for (int b = 0; b < nbTotal; b++) {
      double mx = 0;
      for (int i = 0; i < 32; i++) {
        mx = math.max(mx, x[b * 32 + i].abs());
      }
      final scale = mx / 127.0;
      xscRef[b] = scale;
      final inv = mx > 0 ? 1.0 / scale : 0.0;
      for (int i = 0; i < 32; i++) {
        xqInts[b * 32 + i] =
            (x[b * 32 + i] * inv).roundToDouble().clamp(-127, 127).toInt();
      }
    }
    final bd = ByteData.sublistView(qw8.packed);
    final want = Float64List(T * rows);
    for (int t = 0; t < T; t++) {
      for (int r = 0; r < rows; r++) {
        double acc = 0;
        for (int j = 0; j < nbRow; j++) {
          final base = (r * nbRow + j) * 34;
          final d = halfBitsToFloat(bd.getUint16(base, Endian.little));
          var isum = 0;
          for (int i = 0; i < 32; i++) {
            isum += bd.getInt8(base + 2 + i) * xqInts[(t * nbRow + j) * 32 + i];
          }
          acc += d * xscRef[t * nbRow + j] * isum;
        }
        want[t * rows + r] = acc;
      }
    }

    final wq = gpu.createBuffer(qw8.packed.length, BufferDataType.uint32);
    await wq.write(Uint32List.sublistView(qw8.packed), qw8.packed.length ~/ 4,
        dataType: BufferDataType.uint32);
    final xBuf = gpu.createBuffer(T * cols * 4, BufferDataType.float32);
    await xBuf.write(x, T * cols);
    final xqBuf = gpu.createBuffer(T * cols, BufferDataType.uint32);
    final xscBuf = gpu.createBuffer(nbTotal * 4, BufferDataType.float32);
    final yBuf = gpu.createBuffer(T * rows * 4, BufferDataType.float32);
    final pT = gpu.createBuffer(16, BufferDataType.uint32);
    await pT.write(Uint32List.fromList([T, 0, 0, 0]), 4,
        dataType: BufferDataType.uint32);

    final quant = gpu.createComputeShader()
      ..loadKernelString('''
@group(0) @binding(0) var<storage, read_write> xin: array<f32>;
@group(0) @binding(1) var<storage, read_write> xq: array<u32>;
@group(0) @binding(2) var<storage, read_write> xsc: array<f32>;

var<workgroup> amax: array<f32, 256>;
var<workgroup> qsh: array<i32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let cl: u32 = lid.x / 32u;
  let ln: u32 = lid.x % 32u;
  let blk: u32 = (wid.x + wid.y * nwg.x) * 8u + cl;
  let ok: bool = blk < ${nbTotal}u;
  var v: f32 = 0.0;
  if (ok) { v = xin[blk * 32u + ln]; }
  amax[lid.x] = abs(v);
  workgroupBarrier();
  for (var s: u32 = 16u; s > 0u; s = s >> 1u) {
    if (ln < s) { amax[lid.x] = max(amax[lid.x], amax[lid.x + s]); }
    workgroupBarrier();
  }
  let mx: f32 = amax[cl * 32u];
  let scale: f32 = mx / 127.0;
  let inv: f32 = select(0.0, 1.0 / scale, mx > 0.0);
  qsh[lid.x] = clamp(i32(round(v * inv)), -127, 127);
  workgroupBarrier();
  if (ln < 8u && ok) {
    let b: u32 = cl * 32u + ln * 4u;
    xq[blk * 8u + ln] = (u32(qsh[b]) & 0xFFu) | ((u32(qsh[b + 1u]) & 0xFFu) << 8u)
        | ((u32(qsh[b + 2u]) & 0xFFu) << 16u) | ((u32(qsh[b + 3u]) & 0xFFu) << 24u);
  }
  if (ln == 0u && ok) { xsc[blk] = scale; }
}
''');
    quant.setBuffer('xin', xBuf);
    quant.setBuffer('xq', xqBuf);
    quant.setBuffer('xsc', xscBuf);
    await quant.dispatch((nbTotal + 7) ~/ 8, 1, 1);

    // Dense int8 GEMM — generated exactly as the plan does.
    final jts = List.generate(TB, (i) => i);
    final accDecl = jts.map((i) => '  var acc$i: f32 = 0.0;').join('\n');
    final isumDecl = jts.map((i) => '      var isum$i: i32 = 0;').join('\n');
    final macs = jts
        .map((i) =>
            '        isum$i = isum$i + dot4I8Packed(raw, xsq[xqb + ${i * 8}u + k]);')
        .join('\n');
    final accAdd = jts
        .map((i) =>
            '        acc$i = acc$i + d * xsc[(t0 + ${i}u) * (COLS / 32u) + j] * f32(isum$i);')
        .join('\n');
    final redStore = jts
        .map((i) => '  red[(lid.y * TB + ${i}u) * TPR + lid.x] = acc$i;')
        .join('\n');
    final gemm = gpu.createComputeShader()
      ..loadKernelString('''
@group(0) @binding(0) var<storage, read_write> wq: array<u32>;
@group(0) @binding(1) var<storage, read_write> xq: array<u32>;
@group(0) @binding(2) var<storage, read_write> xsc: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read_write> pT: array<u32>;

const ROWS: u32 = ${rows}u;
const COLS: u32 = ${cols}u;
const TB: u32 = ${TB}u;
const TPR: u32 = ${TPR}u;
const RPW: u32 = ${RPW}u;

${QuantizedTensor.accessorsWGSL}

var<workgroup> xsq: array<u32, ${TPR * 8 * TB}>;
var<workgroup> red: array<f32, ${TB * TPR * RPW}>;

@compute @workgroup_size($TPR, $RPW)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let t0: u32 = wid.y * TB;
  let n: u32 = min(TB, pT[0] - min(t0, pT[0]));
  let tid: u32 = lid.y * TPR + lid.x;
  let row: u32 = wid.x * RPW + lid.y;
  let eb: u32 = 0u;
  let nb: u32 = COLS / 32u;
  let xqb: u32 = lid.x * ${TB * 8}u;
$accDecl
  for (var r: u32 = 0u; r < ${(cols ~/ 32 + TPR - 1) ~/ TPR}u; r = r + 1u) {
    for (var si: u32 = tid; si < ${TPR * 8 * TB}u; si = si + ${TPR * RPW}u) {
      let jj: u32 = r * TPR + si / ${TB * 8}u;
      if (jj < nb) {
        xsq[si] = xq[(t0 + (si / 8u) % TB) * (COLS / 4u) + jj * 8u + (si % 8u)];
      }
    }
    workgroupBarrier();
    let j: u32 = r * TPR + lid.x;
    if (row < ROWS && n > 0u && j < nb) {
      let base: u32 = (row * nb + j) * 34u;
      let d: f32 = f16At(base);
      let qb: u32 = base + 2u;
      let qw: u32 = qb >> 2u;
      let qs: u32 = (qb & 3u) * 8u;
      var carry: u32 = wq[qw];
$isumDecl
      for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        let nxt: u32 = wq[qw + k + 1u];
        var raw: u32;
        if (qs == 0u) { raw = carry; } else { raw = (carry >> qs) | (nxt << (32u - qs)); }
        carry = nxt;
$macs
      }
$accAdd
    }
    workgroupBarrier();
  }
$redStore
  workgroupBarrier();
  for (var pp: u32 = tid; pp < RPW * TB; pp = pp + ${TPR * RPW}u) {
    let rw: u32 = pp / TB;
    let jt: u32 = pp % TB;
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < TPR; i = i + 1u) {
      sum = sum + red[(rw * TB + jt) * TPR + i];
    }
    let orow: u32 = wid.x * RPW + rw;
    if (orow < ROWS && jt < n) {
      y[(t0 + jt) * ROWS + orow] = sum;
    }
  }
}
''');
    gemm.setBuffer('wq', wq);
    gemm.setBuffer('xq', xqBuf);
    gemm.setBuffer('xsc', xscBuf);
    gemm.setBuffer('y', yBuf);
    gemm.setBuffer('pT', pT);
    const rowGroups = (rows + RPW - 1) ~/ RPW;
    const tokTiles = (T + TB - 1) ~/ TB;
    await gemm.dispatch(rowGroups, tokTiles, 1);

    final got = Float32List(T * rows);
    await yBuf.read(got, T * rows);

    double se = 0, ref = 0, maxAbs = 0;
    for (int i = 0; i < T * rows; i++) {
      final d = got[i] - want[i];
      se += d * d;
      ref += want[i] * want[i];
      if (d.abs() > maxAbs) maxAbs = d.abs();
    }
    final relRms = math.sqrt(se / ref);
    // ignore: avoid_print
    print('dense-gemmq relRms: $relRms  maxAbs: $maxAbs');
    expect(relRms, lessThan(1e-5),
        reason: 'dense int8 GEMM diverges (relRms $relRms)');

    quant.destroy();
    gemm.destroy();
    for (final b in [wq, xBuf, xqBuf, xscBuf, yBuf, pT]) {
      b.destroy();
    }
  });
}

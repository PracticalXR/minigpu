import 'dart:typed_data';
import 'dart:math' as math;

import 'package:gpu_tensor/gpu_tensor.dart' show Tensor;
import 'package:minigpu/minigpu.dart';

import 'gguf.dart';
import 'gpu_attn.dart';
import 'gpu_delta_net.dart';
import 'gpu_quant.dart';

/// Fired-dispatch single-token decode plan for qwen35moe (the M5 perf path).
///
/// The v1 runner awaited every dispatch (~3000 completer round trips/token)
/// and rebuilt seqLen-baked shaders every token.  This plan restructures the
/// decode step so that per token there are only ~3 CPU⇄GPU syncs per BLOCK
/// (MoE routing readback) plus one logits readback:
///
/// - All dispatches are fire-and-forget ([ComputeShader.dispatchFire]) and
///   all binds use [ComputeShader.setBufferFire], which joins the same
///   WebGPU-thread FIFO — so bind→fire→rebind→fire sequences are ordered by
///   construction and an awaited readback synchronizes everything queued
///   before it.  Blocks with fully-resident expert stacks are completely
///   sync-free.
/// - Shaders are PLAN-OWNED and shared by source (~60 pipelines total,
///   avoiding the D3D12 many-pipelines failure mode).  Kernels used more
///   than once within a block still get a tag baked into their source
///   (slot0..7, res_attn vs res_ffn, ...) — no longer load-bearing with
///   queued binds, but it keeps per-callsite pipelines distinguishable in
///   captures.
/// - Nothing bakes seqLen: the token position lives in a GPU buffer, the KV
///   cache has fixed capacity, and attention runs as position-driven kernels
///   (prep -> scores -> online-softmax·V) — zero per-token Tint recompiles.
/// - DeltaNet g/beta and MoE softmax/top-k/renormalize run ON the GPU, so the
///   only routing data read back per block is the 8 selected expert ids
///   (needed on the CPU to bind streamed expert weights).
///
/// Correctness relies on two minigpu contracts (see dispatchFire docs):
/// binds mutate shader state immediately while dispatch tasks snapshot
/// bindings when they RUN, and buffer reads/writes join the same FIFO.
class PlanMoe {
  PlanMoe({
    required this.router,
    required this.fetchExpert,
    this.gateShexp,
    this.upShexp,
    this.downShexp,
    this.sharedGate,
    this.stackGate,
    this.stackUp,
    this.stackDown,
  });

  final Tensor router; // [experts, dim] f32
  final Future<QuantizedTensor> Function(String kind, int expert) fetchExpert;
  final QuantizedTensor? gateShexp, upShexp, downShexp;
  final Tensor? sharedGate;

  /// Full-resident [experts, rows, cols] stacks.  When set, this block's MoE
  /// is GPU-DRIVEN: the expert matVec kernels read the winning expert id
  /// straight from the top-k buffer (byte offset = id * bytesPerExpert), so
  /// the block needs NO routing readback and NO expert fetch — it becomes
  /// completely sync-free.
  final QuantizedTensor? stackGate, stackUp, stackDown;

  bool get isResident => stackGate != null;
}

class PlanBlock {
  PlanBlock({
    required this.attnNorm,
    required this.postNorm,
    required this.moe,
    this.attn,
    this.delta,
  });

  final Tensor attnNorm;
  final Tensor postNorm;
  final PlanMoe moe;
  final AttentionLayer? attn;
  final DeltaNetLayer? delta;
}

class _BlockState {
  Buffer? kCache, vCache; // attention blocks
  Buffer? deltaConsts; // [ssmA[vHeads], dtBias[vHeads]] for deltanet blocks
}

class Qwen35DecodePlan {
  Qwen35DecodePlan._({
    required this.gpu,
    required this.blocks,
    required this.outputNorm,
    required this.lmHead,
    required this.dim,
    required this.eps,
    required this.topK,
    required this.maxSeq,
  });

  final Minigpu gpu;
  final List<PlanBlock> blocks;

  /// Final norm + head — null on all but the LAST device of a multi-GPU
  /// pipeline (only the plan that owns the tail produces logits).
  final Tensor? outputNorm;
  final QuantizedTensor? lmHead;
  final int dim;
  final double eps;
  final int topK;
  final int maxSeq;

  int get vocab => lmHead!.rows;

  /// Cumulative microseconds spent awaiting GPU readbacks (routing ids +
  /// logits) — the plan's only sync points.  Benchmarks read deltas.
  int syncMicros = 0;

  // Owned shaders, shared by source.
  final _shaders = <String, ComputeShader>{};
  // matVec shaders memoized by a short key so the (large) source string is
  // only built on first use.
  final _qmvShaders = <String, ComputeShader>{};

  final _blockState = <_BlockState>[];

  // ---- global activation scratch (serialized FIFO execution makes sharing
  // one set across all blocks safe) --------------------------------------
  late final Buffer _x, _xn, _aOut, _ffnOut;
  late final Buffer _pos; // [1] u32, current token position
  late final Buffer? _logits; // only on the plan that owns the head
  // attention
  Buffer? _qFull, _kRaw, _vRaw, _q, _gate, _gated, _scores;
  // deltanet
  Buffer? _qkv, _z, _betaRaw, _alphaRaw, _dParams, _convOut, _qn, _kn, _core,
      _dGated;
  // moe
  late final Buffer _routLogits, _topkIdx, _topkW;
  Buffer? _gExp, _uExp, _prodExp, _gSh, _uSh, _prodSh, _shScalar;
  // fused resident-expert scratch: [topK * rows] stacked over slots.
  Buffer? _gExpAll, _uExpAll, _prodExpAll, _downExpAll;
  // second-stage fusion scratch: [2, topK, rows] (gate plane 0, up plane 1).
  Buffer? _guExpAll;
  // chain-fuse scratch: shexp [gate rows; up rows], delta [qkv rows; z rows].
  Buffer? _guSh, _qkvz;
  // resid-fuse scratch: the shared expert's gated output, folded into the
  // MoE combine's single write into x (so no ffnOut zero pass is needed).
  Buffer? _shOut;
  // int8-quantized activations for the dp4a decode path: packed int8 (u32)
  // + per-32-block f32 scales.  _xnq = quantized _xn (shared by all
  // projections that read the normed hidden state); _prodq = quantized
  // silu.mul expert intermediates (per-slot).
  Buffer? _xnq, _xnsc, _prodq, _prodsc;

  static Future<Qwen35DecodePlan> build({
    required Minigpu gpu,
    required List<PlanBlock> blocks,
    Tensor? outputNorm,
    QuantizedTensor? lmHead,
    required int dim,
    required double eps,
    required int topK,
    int maxSeq = 4096,
  }) async {
    final p = Qwen35DecodePlan._(
      gpu: gpu,
      blocks: blocks,
      outputNorm: outputNorm,
      lmHead: lmHead,
      dim: dim,
      eps: eps,
      topK: topK,
      maxSeq: maxSeq,
    );
    await p._init();
    return p;
  }

  Buffer _f32(int elems) =>
      gpu.createBuffer(elems * 4, BufferDataType.float32);
  Buffer _u32(int elems) => gpu.createBuffer(elems * 4, BufferDataType.uint32);

  Future<void> _init() async {
    _x = _f32(dim);
    _xn = _f32(dim);
    _aOut = _f32(dim);
    _ffnOut = _f32(dim);
    _pos = _u32(4); // min binding sizes are generous; 4 words is fine
    _logits = lmHead != null ? _f32(vocab) : null;

    final attn = blocks.map((b) => b.attn).whereType<AttentionLayer>();
    if (attn.isNotEmpty) {
      final a = attn.first;
      _qFull = _f32(a.wq.rows);
      _kRaw = _f32(a.kvDim);
      _vRaw = _f32(a.kvDim);
      _q = _f32(a.qDim);
      _gate = _f32(a.qDim);
      _gated = _f32(a.qDim);
      _scores = _f32(a.heads * maxSeq);
    }
    final delta = blocks.map((b) => b.delta).whereType<DeltaNetLayer>();
    if (delta.isNotEmpty) {
      final d = delta.first;
      _qkv = _f32(d.convDim);
      _z = _f32(d.valueDim);
      _betaRaw = _f32(d.vHeads);
      _alphaRaw = _f32(d.vHeads);
      _dParams = _f32(2 * d.vHeads);
      _convOut = _f32(d.convDim);
      _qn = _f32(d.keyDim);
      _kn = _f32(d.keyDim);
      _core = _f32(d.valueDim);
      _dGated = _f32(d.valueDim);
    }
    final experts = blocks.first.moe.router.shape[0];
    _routLogits = _f32(experts);
    _topkIdx = _u32(topK);
    _topkW = _f32(topK);

    for (final b in blocks) {
      final st = _BlockState();
      if (b.attn != null) {
        st.kCache = _f32(maxSeq * b.attn!.kvDim);
        st.vCache = _f32(maxSeq * b.attn!.kvDim);
      }
      if (b.delta != null) {
        final d = b.delta!;
        st.deltaConsts = _f32(2 * d.vHeads);
        final consts = Float32List(2 * d.vHeads);
        consts.setRange(0, d.vHeads, d.ssmA);
        consts.setRange(d.vHeads, 2 * d.vHeads, d.dtBias);
        await st.deltaConsts!.write(consts, 2 * d.vHeads);
      }
      final sh = b.moe.gateShexp;
      if (sh != null && _gSh == null) {
        _gSh = _f32(sh.rows);
        _uSh = _f32(sh.rows);
        _prodSh = _f32(sh.rows);
        _shScalar = _f32(4);
      }
      _blockState.add(st);
    }
  }

  // ---- shader helpers ----------------------------------------------------

  ComputeShader _sh(String src) => _shaders.putIfAbsent(src, () {
        final s = gpu.createComputeShader();
        s.loadKernelString(src);
        return s;
      });

  static String _f(double v) {
    final s = v.toString();
    return (s.contains('.') || s.contains('e') || s.contains('E'))
        ? s
        : '$s.0';
  }

  void _fireRows(ComputeShader s, int rows) {
    final x = rows <= 65535 ? rows : 65535;
    final y = (rows + x - 1) ~/ x;
    s.dispatchFire(x == 0 ? 1 : x, y == 0 ? 1 : y, 1);
  }

  /// Dispatch one workgroup per (row, z) with rows folded over x/y and z on
  /// the third dim — used to run all top-K expert slots in ONE dispatch.
  void _fireRowsZ(ComputeShader s, int rows, int z) {
    final x = rows <= 65535 ? rows : 65535;
    final y = (rows + x - 1) ~/ x;
    s.dispatchFire(x == 0 ? 1 : x, y == 0 ? 1 : y, z);
  }

  void _fireLinear(ComputeShader s, int threads) {
    final wg = (threads + 255) ~/ 256;
    final x = wg <= 65535 ? wg : 65535;
    final y = (wg + x - 1) ~/ x;
    s.dispatchFire(x == 0 ? 1 : x, y == 0 ? 1 : y, 1);
  }

  static String _reduceSum(int wg) => '''
  workgroupBarrier();
  for (var s: u32 = ${wg ~/ 2}u; s > 0u; s = s >> 1u) {
    if (lid.x < s) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s]; }
    workgroupBarrier();
  }
''';

  /// Threads-per-row for a single-token decode GEMV: the largest power of
  /// two <= the block count (COLS/blockSize) and <= 64, floored at 8, so
  /// 256/T rows pack one workgroup with ~full lane occupancy.  q8_0 COLS
  /// 2048 -> 64 blocks -> T=64 (R=4); down COLS ~768 -> 24 blocks -> T=16
  /// (R=16).  The old one-row-per-256-lane layout idled 75-90% of lanes and
  /// ran an 8-step reduction over mostly zeros.
  static int _gemvT(int cols, int rows, int type) {
    if (const bool.fromEnvironment('GPU_ML_SLOW_GEMV')) return 256;
    // Row-packing (T<256, R=256/T rows per workgroup) gives full lane
    // occupancy but FEWER workgroups — a net loss for tiny-output GEMVs
    // (e.g. DeltaNet beta/alpha, ~32 rows) that are already workgroup-starved.
    // Only pack when rows are plentiful enough to keep the GPU busy.
    if (rows < 512) return 256;
    final bs = ggmlTypeTraits[type]!.blockSize;
    final nb = cols ~/ bs;
    var t = 64;
    while (t > 8 && t > nb) {
      t ~/= 2;
    }
    return t;
  }

  /// Threads-per-row for the ROW-SPLIT dp4a variant: enough rows per
  /// workgroup for occupancy while keeping >=2 serial blocks per thread so
  /// the int8 MAC stream is long enough to matter.
  static int _dp4aT(int cols) {
    final nb = cols ~/ 32;
    var t = 16;
    while (t > 4 && t > nb ~/ 2) {
      t ~/= 2;
    }
    return t;
  }

  /// dot4I8Packed matVec for a q8_0 weight against pre-quantized int8
  /// activations ([xq]/[xsc]).  SHAPE-AWARE structure:
  ///  * rows >= 64k (lm_head): THREAD-PER-ROW, no reduction — hundreds of
  ///    workgroups, longest serial stream per thread.
  ///  * smaller projections: ROW-SPLIT (T threads/row x 256/T rows/wg,
  ///    T-wide reduction) — the pure thread-per-row form starved these to
  ///    3-16 workgroups (~2% GPU util) and REGRESSED decode ~30%.
  void _fireQmvDp4a(
      QuantizedTensor w, String tag, Buffer xq, Buffer xsc, Buffer y) {
    if (w.type != GgmlType.q8_0) {
      throw StateError('_fireQmvDp4a requires q8_0');
    }
    final tpr = w.rows >= 65536 ? 1 : _dp4aT(w.cols);
    final xs = tpr == 1 && _sharedXOk(w.cols);
    final key = 'dq/$tag/${w.rows}x${w.cols}/T$tpr${xs ? '/xs' : ''}';
    final s = _qmvShaders.putIfAbsent(key, () => _sh(tpr == 1
        ? '''
// plan-qmvq:$tag${xs ? ':xs' : ''}
@group(0) @binding(0) var<storage, read_write> wq: array<u32>;
@group(0) @binding(1) var<storage, read_write> xq: array<u32>;
@group(0) @binding(2) var<storage, read_write> xsc: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

const ROWS: u32 = ${w.rows}u;
const COLS: u32 = ${w.cols}u;

${QuantizedTensor.accessorsWGSL}
${xs ? '''
var<workgroup> xqs: array<u32, ${w.cols ~/ 4}>;
var<workgroup> xscs: array<f32, ${w.cols ~/ 32}>;
''' : ''}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let row: u32 = (wid.x + wid.y * nwg.x) * 256u + lid.x;
  let eb: u32 = 0u;
  var acc: f32 = 0.0;
${xs ? '''
  for (var ix: u32 = lid.x; ix < ${w.cols ~/ 4}u; ix = ix + 256u) { xqs[ix] = xq[ix]; }
  for (var ix: u32 = lid.x; ix < ${w.cols ~/ 32}u; ix = ix + 256u) { xscs[ix] = xsc[ix]; }
  workgroupBarrier();
''' : ''}
  if (row < ROWS) {
${xs ? QuantizedTensor.matVecDp4aBodyWGSL(threadVar: '0u', stride: '1u').replaceAll('xsc[', 'xscs[').replaceAll('xq[', 'xqs[') : QuantizedTensor.matVecDp4aBodyWGSL(threadVar: '0u', stride: '1u')}
  }
  if (row < ROWS) { y[row] = acc; }
}
'''
        : '''
// plan-qmvq:$tag:T$tpr
@group(0) @binding(0) var<storage, read_write> wq: array<u32>;
@group(0) @binding(1) var<storage, read_write> xq: array<u32>;
@group(0) @binding(2) var<storage, read_write> xsc: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

const ROWS: u32 = ${w.rows}u;
const COLS: u32 = ${w.cols}u;

${QuantizedTensor.accessorsWGSL}

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let trd: u32 = lid.x % ${tpr}u;
  let row: u32 = (wid.x + wid.y * nwg.x) * ${256 ~/ tpr}u + lid.x / ${tpr}u;
  let eb: u32 = 0u;
  var acc: f32 = 0.0;
  if (row < ROWS) {
${QuantizedTensor.matVecDp4aBodyWGSL(threadVar: 'trd', stride: '${tpr}u')}
  }
  scratch[lid.x] = acc;
  workgroupBarrier();
  for (var s: u32 = ${tpr ~/ 2}u; s > 0u; s = s >> 1u) {
    if (trd < s) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s]; }
    workgroupBarrier();
  }
  if (trd == 0u && row < ROWS) { y[row] = scratch[lid.x]; }
}
'''));
    s.setBuffer('wq', w.buffer);
    s.setBuffer('xq', xq);
    s.setBuffer('xsc', xsc);
    s.setBuffer('y', y);
    final rowsPerWg = tpr == 1 ? 256 : 256 ~/ tpr;
    final g = (w.rows + rowsPerWg - 1) ~/ rowsPerWg;
    final gx = g <= 65535 ? g : 65535;
    s.dispatchFire(gx, (g + gx - 1) ~/ gx, 1);
  }

  /// True when q8_0 dense projections should take the dp4a path.
  bool _dp4aOk(QuantizedTensor w) => _dp4aExperts && w.type == GgmlType.q8_0;

  /// Fused-dequant matVec, fired.  Shaders are memoized per
  /// (tag, rows, cols, type); tags keep within-block instances distinct.
  void _fireQmv(QuantizedTensor w, String tag, Buffer x, Buffer y) {
    final xs = _sharedXOk(w.cols);
    final t = xs
        ? _gemvTXs(w.cols, w.rows, w.type)
        : _gemvT(w.cols, w.rows, w.type);
    final key = '$tag/${w.rows}x${w.cols}/t${w.type}/T$t${xs ? '/xs' : ''}';
    final s = _qmvShaders.putIfAbsent(
        key,
        () => _sh(_qmvSrc(w.rows, w.cols, w.type, tag,
            threadsPerRow: t, sharedX: xs)));
    s.setBuffer('wq', w.buffer);
    s.setBuffer('x', x);
    s.setBuffer('y', y);
    _fireRows(s, (w.rows + 256 ~/ t - 1) ~/ (256 ~/ t));
  }

  /// matVec variant that ACCUMULATES `y[row] += wsel[slot] * dot` — used for
  /// per-expert weighted combine and the shared-expert sigmoid gate.
  void _fireQmvAccum(
      QuantizedTensor w, String tag, Buffer x, Buffer y, Buffer wsel,
      int slot) {
    final xs = _sharedXOk(w.cols);
    final t = xs
        ? _gemvTXs(w.cols, w.rows, w.type)
        : _gemvT(w.cols, w.rows, w.type);
    final key =
        'acc$slot/$tag/${w.rows}x${w.cols}/t${w.type}/T$t${xs ? '/xs' : ''}';
    final s = _qmvShaders.putIfAbsent(
        key,
        () => _sh(_qmvSrc(w.rows, w.cols, w.type, tag,
            accumSlot: slot, threadsPerRow: t, sharedX: xs)));
    s.setBuffer('wq', w.buffer);
    s.setBuffer('x', x);
    s.setBuffer('y', y);
    s.setBuffer('wsel', wsel);
    _fireRows(s, (w.rows + 256 ~/ t - 1) ~/ (256 ~/ t));
  }

  /// TWO q8_0 weights (same cols) against the same x in ONE dispatch: rows
  /// CONCATENATED into [y] (w1's rows first).  Row-range branch a la
  /// plan-dnbap; the second weight gets renamed accessors.  Requires the
  /// shared-x contract; callers fall back to two _fireQmv otherwise.
  void _fireQmvCat2(QuantizedTensor w1, QuantizedTensor w2, String tag,
      Buffer x, Buffer y) {
    final rt = w1.rows + w2.rows;
    final t = _gemvTXs(w1.cols, rt, w1.type);
    final r = 256 ~/ t;
    // The type MUST be in the key: blocks with identical shapes come in
    // both q8_0 and f16 in this build, and the bodies differ.
    final key =
        'cat2/$tag/${w1.rows}+${w2.rows}x${w1.cols}/t${w1.type}/T$t';
    final s = _qmvShaders.putIfAbsent(key, () {
      final accB = QuantizedTensor.accessorsWGSL
          .replaceAll('wq', 'wqb')
          .replaceAll('f16At', 'bf16At')
          .replaceAll('byteAt', 'bbyteAt')
          .replaceAll('wAt', 'bwAt');
      var bodyA = QuantizedTensor.matVecBodyWGSL(w1.type,
              threadVar: 'trd', stride: '${t}u')
          .replaceAll('row *', 'rEff *');
      bodyA = QuantizedTensor.sharedXBody(bodyA);
      final bodyB =
          bodyA.replaceAll('wq[', 'wqb[').replaceAll('f16At(', 'bf16At(');
      return _sh('''
// plan-qmvcat2:$tag:T$t
@group(0) @binding(0) var<storage, read_write> wq: array<u32>;
@group(0) @binding(1) var<storage, read_write> wqb: array<u32>;
@group(0) @binding(2) var<storage, read_write> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

const R1: u32 = ${w1.rows}u;
const RT: u32 = ${rt}u;
const COLS: u32 = ${w1.cols}u;

${QuantizedTensor.accessorsWGSL}
$accB
${QuantizedTensor.xsDeclWGSL(w1.cols)}

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  ${t == 256 ? '' : 'let trd: u32 = lid.x % ${t}u;'}
  let row: u32 = ${t == 256 ? 'wid.x + wid.y * nwg.x' : '(wid.x + wid.y * nwg.x) * ${r}u + lid.x / ${t}u'};
  let eb: u32 = 0u;
  var acc: f32 = 0.0;
${QuantizedTensor.stageXsWGSL(w1.cols)}
  if (row < RT) {
    let isA: bool = row < R1;
    let rEff: u32 = select(row - R1, row, isA);
    if (isA) {
$bodyA
    } else {
$bodyB
    }
  }
  scratch[lid.x] = acc;
  workgroupBarrier();
  for (var s: u32 = ${(t == 256 ? 256 : t) ~/ 2}u; s > 0u; s = s >> 1u) {
    if (${t == 256 ? 'lid.x' : 'trd'} < s) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s]; }
    workgroupBarrier();
  }
  if (${t == 256 ? 'lid.x == 0u' : 'trd == 0u'} && row < RT) {
    y[row] = scratch[lid.x];
  }
}
''');
    });
    s.setBuffer('wq', w1.buffer);
    s.setBuffer('wqb', w2.buffer);
    s.setBuffer('x', x);
    s.setBuffer('y', y);
    _fireRows(s, (rt + r - 1) ~/ r);
  }

  /// Types the fused/concatenated decode kernels can generate bodies for.
  /// This imatrix build keeps 77 tensors at f16 — 13 blocks' expert stacks,
  /// 12 shexp pairs, some qkv/gate/alpha/beta — so gating the fusions on
  /// q8_0 silently dropped a third of the model onto the unfused paths.
  static bool _fusableType(int t) => t == GgmlType.q8_0 || t == GgmlType.f16;

  /// True when [_fireQmvCat2] can take a weight pair.  Both weights share
  /// one generated body, so they must share a type.
  bool _cat2Ok(QuantizedTensor w1, QuantizedTensor w2) =>
      chainFuse &&
      xsRowPack &&
      _fusableType(w1.type) &&
      w1.type == w2.type &&
      w1.cols == w2.cols &&
      _sharedXOk(w1.cols);

  /// Output projection: with [residFuse] the dot accumulates straight into
  /// x (the residual add becomes free); otherwise it writes _aOut for the
  /// separate add dispatch.
  void _fireOutProj(QuantizedTensor w, String tag, Buffer src) {
    if (!residFuse) {
      _fireQmv(w, tag, src, _aOut);
      return;
    }
    final xs = _sharedXOk(w.cols);
    final t = xs
        ? _gemvTXs(w.cols, w.rows, w.type)
        : _gemvT(w.cols, w.rows, w.type);
    final key = 'addx/$tag/${w.rows}x${w.cols}/t${w.type}/T$t${xs ? '/xs' : ''}';
    final s = _qmvShaders.putIfAbsent(
        key,
        () => _sh(_qmvSrc(w.rows, w.cols, w.type, tag,
            threadsPerRow: t, sharedX: xs, accumPlain: true)));
    s.setBuffer('wq', w.buffer);
    s.setBuffer('x', src);
    s.setBuffer('y', _x);
    _fireRows(s, (w.rows + 256 ~/ t - 1) ~/ (256 ~/ t));
  }

  String _qmvSrc(int rows, int cols, int type, String tag,
      {int? accumSlot,
      int threadsPerRow = 256,
      bool sharedX = false,
      bool accumPlain = false}) {
    final accum = accumSlot != null;
    final t = threadsPerRow;
    final r = 256 ~/ t;
    var body = t == 256
        ? QuantizedTensor.matVecBodyWGSL(type)
        : QuantizedTensor.matVecBodyWGSL(type,
            threadVar: 'trd', stride: '${t}u');
    if (sharedX) body = QuantizedTensor.sharedXBody(body);
    // Reduce within each row's T-lane group: scratch[rr*T + t], tree over t.
    final reduce = t == 256
        ? _reduceSum(256)
        : '''
  workgroupBarrier();
  for (var s: u32 = ${t ~/ 2}u; s > 0u; s = s >> 1u) {
    if (trd < s) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s]; }
    workgroupBarrier();
  }
''';
    final rowExpr = t == 256
        ? 'wid.x + wid.y * nwg.x'
        : '(wid.x + wid.y * nwg.x) * ${r}u + lid.x / ${t}u';
    final firstLane = t == 256 ? 'lid.x == 0u' : 'trd == 0u';
    return '''
// plan-qmv:$tag${t == 256 ? '' : ':T$t'}${sharedX ? ':xs' : ''}
@group(0) @binding(0) var<storage, read_write> wq: array<u32>;
@group(0) @binding(1) var<storage, read_write> x: array<${sharedX ? 'vec4<f32>' : 'f32'}>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
${accum ? '@group(0) @binding(3) var<storage, read_write> wsel: array<f32>;' : ''}
${sharedX ? QuantizedTensor.xsDeclWGSL(cols) : ''}

const ROWS: u32 = ${rows}u;
const COLS: u32 = ${cols}u;

${QuantizedTensor.accessorsWGSL}
${QuantizedTensor.typeNeedsScaleMinK4(type) ? QuantizedTensor.scaleMinK4WGSL : ''}

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  ${t == 256 ? '' : 'let trd: u32 = lid.x % ${t}u;'}
  let row: u32 = $rowExpr;
  let eb: u32 = 0u;
  var acc: f32 = 0.0;
${sharedX ? QuantizedTensor.stageXsWGSL(cols) : ''}
  if (row < ROWS) {
$body
  }
  scratch[lid.x] = acc;
$reduce
  if ($firstLane && row < ROWS) {
    ${accum ? 'y[row] = y[row] + wsel[${accumSlot}u] * scratch[lid.x];' : accumPlain ? 'y[row] = y[row] + scratch[lid.x];' : 'y[row] = scratch[lid.x];'}
  }
}
''';
  }

  /// Fires the per-32-block int8 activation quantizer over [nElems] f32 of
  /// [src] into [xq] (packed int8) + [xsc] (block scales).  One 32-thread
  /// workgroup per block.  Used to feed the dp4a matVec path.
  void _fireQuantX(String tag, Buffer src, int nElems, Buffer xq, Buffer xsc) {
    final nb = nElems ~/ 32;
    final s = _sh('''
// plan-quantx:$tag
@group(0) @binding(0) var<storage, read_write> xin: array<f32>;
@group(0) @binding(1) var<storage, read_write> xq: array<u32>;
@group(0) @binding(2) var<storage, read_write> xsc: array<f32>;
${QuantizedTensor.quantizeInt8BlockWGSL}
@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let blk: u32 = wid.x + wid.y * nwg.x;
  if (blk >= ${nb}u) { return; }
  quantizeBlock(blk, 0u, lid.x);
}
''');
    s.setBuffer('xin', src);
    s.setBuffer('xq', xq);
    s.setBuffer('xsc', xsc);
    _fireRows(s, nb);
  }

  /// BULK int8 activation quantizer for prefill-sized inputs: 256-thread
  /// workgroups quantize 8 blocks each (the decode variant's one-block
  /// 32-thread workgroups would launch ~100k starved workgroups on a
  /// T*topK*rows intermediate).  All clusters share the workgroup barriers
  /// uniformly; out-of-range clusters compute on zeros and skip writes.
  void _fireQuantXB(String tag, Buffer src, int nElems, Buffer xq, Buffer xsc) {
    final nb = nElems ~/ 32;
    final s = _sh('''
// plan-quantxb:$tag
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
  let ok: bool = blk < ${nb}u;
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
    s.setBuffer('xin', src);
    s.setBuffer('xq', xq);
    s.setBuffer('xsc', xsc);
    _fireRows(s, (nb + 7) ~/ 8);
  }

  /// dot4I8Packed decode level: 0 = off (f32 everywhere), 1 = lm_head only
  /// (the one huge streaming matVec, where the int8 MAC saving is
  /// microbench-proven and costs a single extra quantize), 2 = all q8_0
  /// projections (experts/shexp/delta/attn — each group adds a tiny
  /// latency-bound quantize dispatch, which in-process A/B showed eating
  /// the savings).  Numerics validated at every level (relRms 0.0038,
  /// decode text byte-identical).  RUNTIME-switchable so a bench can
  /// alternate levels within one process — the only fair A/B on a machine
  /// with other GPU load.  Seeded from -DGPU_ML_DP4A (=2).
  int dp4aLevel = const bool.fromEnvironment('GPU_ML_DP4A') ? 2 : 0;

  set dp4aEnabled(bool v) => dp4aLevel = v ? 2 : 0;

  bool get _dp4aExperts => dp4aLevel >= 2;

  bool get _dp4aHead => dp4aLevel >= 1 && lmHead?.type == GgmlType.q8_0;

  /// dot4I8Packed in the PREFILL grouped expert kernels.  Unlike decode,
  /// these are arithmetic-heavy (TB=16 tokens per weight word = 16 MACs per
  /// weight byte), the regime where the int8 MAC saving is microbench-
  /// proven.  DEFAULT ON: fair interleaved A/B measured +10% prefill
  /// (277.6 -> 305.5 tok/s @512, 3/3 non-overlapping pairs); numerics in
  /// the accepted class (e2e relRms 0.029 incl. the f16 dense tradeoff,
  /// top-5 identical, greedy text byte-identical).  RUNTIME-switchable;
  /// -DGPU_ML_NO_PREFILL_DP4A opts out.
  bool prefillDp4a = !const bool.fromEnvironment('GPU_ML_NO_PREFILL_DP4A');

  /// DeltaNet decode fusion: beta+alpha GEMVs + the params transform run as
  /// ONE kernel (3 dispatches -> 1), and the gated norm folds into the
  /// recurrence kernel's epilogue (same vHeads x headDim grid; the fused
  /// variant writes the gated output directly, skipping the core
  /// round-trip).  ~90 fewer dispatches/token across 30 delta blocks.
  /// DEFAULT ON: math is order-identical to the unfused path; fair
  /// interleaved A/B measured +3% decode (34.07 -> 35.06 tok/s).
  /// RUNTIME-switchable; -DGPU_ML_NO_DELTA_FUSE opts out.
  bool deltaFusion = !const bool.fromEnvironment('GPU_ML_NO_DELTA_FUSE');

  /// Shared-memory x staging for decode matVecs: D3D11/FXC storage-buffer
  /// reads of the activation vector inside the dot loop stall it ~4x
  /// (kernel-tax probe, wave 15); staging x into a padded workgroup array
  /// once per workgroup runs 2.8x faster at identical math (bit-exact —
  /// only the load source changes).  RUNTIME-switchable (distinct memo
  /// keys); -DGPU_ML_NO_SHARED_X opts out.
  bool sharedX = !const bool.fromEnvironment('GPU_ML_NO_SHARED_X');

  /// Probe-validated per-shape T for shared-x kernels
  /// (tool/xs_shape_probe.dart): dense 2048x2048 T=32 (612 GB/s vs 558 at
  /// T=64); expert gate 512x2048x8 T=16 (963 vs 385 at the old T=256!);
  /// expert down 2048x512x8 T=16 (829 vs 117 at T=256 — 240 of 256 lanes
  /// idle on a 16-block row plus 33 MB staged per 8.9 MB of weights).
  /// NOTE the first, heuristic retune (wgs>=96 rule -> T=8) measured
  /// SLOWER e2e — T=8's serial loops overshoot the sweet spot; only the
  /// probe table is trustworthy.  RUNTIME-switchable (--ab-rp).
  bool xsRowPack = !const bool.fromEnvironment('GPU_ML_NO_XS_ROWPACK');

  /// Shared-x applies when the staging contract holds: vec4 loads need
  /// cols % 4 == 0, and the padded array must fit workgroup storage
  /// alongside the 1 KB reduction scratch (cols <= 4096 -> <= ~17 KB).
  bool _sharedXOk(int cols) => sharedX && cols % 4 == 0 && cols <= 4096;

  /// Threads-per-row for SHARED-X GEMVs.  Staging costs cols*4 bytes per
  /// WORKGROUP, so rows/wg (256/T) wants to be as high as occupancy allows:
  /// pick the smallest T (most rows/wg, least staging) that still launches
  /// >= ~96 workgroups (the occupancy probe knee: 64 wg = 1570 GB/s,
  /// 128 wg = 1690).  At T=16 a 2048-row GEMV stages 128 x 8 KB = 1 MB vs
  /// 4.3 MB of weights (vs 4.2 MB staged at the non-xs T=64 tuning).
  int _gemvTXs(int cols, int rows, int type) {
    if (!xsRowPack) return _gemvT(cols, rows, type);
    if (rows < 512) return 256; // tiny-output GEMVs stay one-row-per-wg
    final bs = ggmlTypeTraits[type]!.blockSize;
    final nb = cols ~/ bs;
    if (nb >= 32) return 32; // probe winner for dense residual projections
    return _gemvT(cols, rows, type);
  }

  /// Probe-validated T for the slotted expert kernels (see [xsRowPack]).
  int _expTXs(int cols, int type) {
    final nb = cols ~/ ggmlTypeTraits[type]!.blockSize;
    return nb >= 16 ? 16 : 256;
  }

  /// Second-stage expert fusion: gate+up in ONE dispatch (z = 2*topK, dual
  /// weight bindings a la plan-dnbap) and silu(gate)*up folded into the
  /// down kernel's shared-x staging loop (kills the silumul dispatch + the
  /// prod buffer round-trip).  5 dispatches/block -> 3.  Motivated by the
  /// overlap probe: the backend serializes ALL dispatches (disjoint-buffer
  /// pair costs sum-of-solos, D3D11 AND D3D12) — fewer/bigger kernels is
  /// the only way to buy back per-dispatch latency.  Bit-exact: identical
  /// expressions and accumulation order.  RUNTIME-switchable (--ab-f2).
  bool expFuse2 = !const bool.fromEnvironment('GPU_ML_NO_EXP_FUSE2');

  /// Third fusion stage: shexp gate+up as ONE concatenated-rows dispatch +
  /// silu·mul folded into sh_down's staging (5 -> 3 dispatches), and delta
  /// qkv+z projections as ONE concatenated dispatch (rec reads z at a baked
  /// offset).  Bit-exact.  RUNTIME-switchable (--ab-cf).
  bool chainFuse = !const bool.fromEnvironment('GPU_ML_NO_CHAIN_FUSE');

  /// Residual folding: the attn/delta OUTPUT projection accumulates straight
  /// into x (`y[row] += dot`) instead of writing _aOut for a separate add,
  /// and the MoE combine writes `x += shexp + sum_slot` directly — which
  /// also removes the ffnOut zero pass (the prefill combine already works
  /// this way).  Kills 3 tiny dispatches per block (~120/token), each of
  /// which costs a full serialized launch (~5 us) for 8 KB of work.
  /// Bit-exact: same expressions, same order.  RUNTIME-switchable (--ab-rf).
  bool residFuse = !const bool.fromEnvironment('GPU_ML_NO_RESID_FUSE');

  /// DeltaNet recurrence occupancy split: the fused rec kernel runs ONE
  /// workgroup per v-head (~32) — only ~25% of the 4090's SMs can be busy no
  /// matter how big the workgroup is, and it moves the most memory of any
  /// decode kernel (state = vHeads*D*D f32, read+written twice per token).
  /// This splits each head's rows across several workgroups (TPR lanes per
  /// row + a TPR-wide reduction), which forces the gated-norm epilogue back
  /// out into its own dispatch (the norm reduces over ALL rows of a head).
  /// NOT bit-exact: the row dot products become tree sums instead of serial
  /// sums.  RUNTIME-switchable (--ab-rs).
  bool recSplit = !const bool.fromEnvironment('GPU_ML_NO_REC_SPLIT');

  /// Timing-bisect ablations, RUNTIME so a whole category map can be taken
  /// inside ONE process (`--ab-map`): cross-process comparisons on this
  /// machine are worthless (0.8-45 tok/s swings), so a map built from one
  /// process per category measured mostly noise.  Output is garbage while
  /// any of these is set — bench with --ignore-eos and read only the
  /// per-level ms/token.  Env defines keep the old compile-time names.
  bool skipAttn = const bool.fromEnvironment('GPU_ML_DEC_NO_ATTN');
  bool skipMoe = const bool.fromEnvironment('GPU_ML_DEC_NO_MOE');
  bool skipDelta = const bool.fromEnvironment('GPU_ML_DEC_NO_DELTA');
  bool skipAttnOnly = const bool.fromEnvironment('GPU_ML_DEC_NO_ATTNONLY');
  bool skipShexp = const bool.fromEnvironment('GPU_ML_DEC_NO_SHEXP');
  bool skipExp = const bool.fromEnvironment('GPU_ML_DEC_NO_EXP');
  bool skipProj = const bool.fromEnvironment('GPU_ML_DN_NO_PROJ');
  bool skipBap = const bool.fromEnvironment('GPU_ML_DN_NO_BAP');
  bool skipConv = const bool.fromEnvironment('GPU_ML_DN_NO_CONV');
  bool skipRec = const bool.fromEnvironment('GPU_ML_DN_NO_REC2') ||
      const bool.fromEnvironment('GPU_ML_DEC_NO_REC');
  bool skipOut = const bool.fromEnvironment('GPU_ML_DN_NO_OUT');
  /// Skips both per-block RMS norms (the two remaining SINGLE-workgroup
  /// dispatches per block) — measures whether folding them into their
  /// consumers is worth the plumbing.
  bool skipNorms = false;
  /// Skips the router GEMV + top-k select.  The expert kernels then reuse
  /// the ids the last unskipped token wrote, so indices stay in range.
  bool skipRoute = false;

  /// Lanes per state row for [recSplit]: the smallest split that puts the
  /// dispatch past the occupancy knee measured by tool/occupancy_probe.dart
  /// (64 wg = 1570 GB/s, 128 wg = 1690, vs 32 wg = 852).
  static int _recTpr(int hd, int vHeads) {
    for (final t in const [1, 2, 4, 8]) {
      final rpw = 256 ~/ t;
      if (vHeads * ((hd + rpw - 1) ~/ rpw) >= 128) return t;
    }
    return 8;
  }

  /// Runs ALL top-K resident experts in 5 dispatches (gate, up, silu·mul,
  /// down, weighted-combine) instead of 4 per slot.  Each matVec dispatch
  /// covers every slot at once via `wid.z = slot`, reading the winning
  /// expert id from the top-k buffer — the per-token dispatch count is the
  /// main decode-time cost once weights are resident.  q8_0 stacks take the
  /// dot4I8Packed path (int8 activations); others fall back to f32.
  void _fireExpertsFused(PlanMoe m,
      {bool directX = false, bool hasSh = false}) {
    final gk = m.stackGate!, uk = m.stackUp!, dk = m.stackDown!;
    final interRows = gk.rows; // expert intermediate dim
    _gExpAll ??= _f32(topK * interRows);
    _uExpAll ??= _f32(topK * interRows);
    _prodExpAll ??= _f32(topK * interRows);
    _downExpAll ??= _f32(topK * dim);

    final dp4a = _dp4aExperts &&
        gk.type == GgmlType.q8_0 &&
        uk.type == GgmlType.q8_0 &&
        dk.type == GgmlType.q8_0;
    if (dp4a) {
      // _xnq/_xnsc are populated once by the caller (_moe) before shexp.
      _prodq ??= _u32(topK * interRows ~/ 4);
      _prodsc ??= _f32(topK * interRows ~/ 32);
    }

    // gate / up: y[slot*ROWS + row] = W[idxb[slot]] @ x, all slots at once.
    void fusedMatVec(QuantizedTensor stack, String tag, Buffer x, Buffer y) {
      final traits = ggmlTypeTraits[stack.type]!;
      final bpe = stack.rows * stack.cols ~/ traits.blockSize * traits.typeSize;
      final xs = _sharedXOk(stack.cols);
      // Shared-x pays staging (cols*4 B) per WORKGROUP: pack R=256/T rows
      // per wg, sizing T by the wgs actually launched across all z-slots.
      final t = xs && xsRowPack ? _expTXs(stack.cols, stack.type) : 256;
      final r = 256 ~/ t;
      var body = t == 256
          ? QuantizedTensor.matVecBodyWGSL(stack.type)
          : QuantizedTensor.matVecBodyWGSL(stack.type,
              threadVar: 'trd', stride: '${t}u');
      if (xs) body = QuantizedTensor.sharedXBody(body);
      final reduce = t == 256
          ? _reduceSum(256)
          : '''
  workgroupBarrier();
  for (var s: u32 = ${t ~/ 2}u; s > 0u; s = s >> 1u) {
    if (trd < s) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s]; }
    workgroupBarrier();
  }
''';
      final s = _sh('''
// plan-expfused:$tag${xs ? ':xs' : ''}${t == 256 ? '' : ':T$t'}
@group(0) @binding(0) var<storage, read_write> wq: array<u32>;
@group(0) @binding(1) var<storage, read_write> x: array<${xs ? 'vec4<f32>' : 'f32'}>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> idxb: array<u32>;

const ROWS: u32 = ${stack.rows}u;
const COLS: u32 = ${stack.cols}u;

${QuantizedTensor.accessorsWGSL}
${QuantizedTensor.typeNeedsScaleMinK4(stack.type) ? QuantizedTensor.scaleMinK4WGSL : ''}
${xs ? QuantizedTensor.xsDeclWGSL(stack.cols) : ''}

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let slot: u32 = wid.z;
  ${t == 256 ? '' : 'let trd: u32 = lid.x % ${t}u;'}
  let row: u32 = ${t == 256 ? 'wid.x + wid.y * nwg.x' : '(wid.x + wid.y * nwg.x) * ${r}u + lid.x / ${t}u'};
  let eb: u32 = idxb[slot] * ${bpe}u;
  var acc: f32 = 0.0;
${xs ? QuantizedTensor.stageXsWGSL(stack.cols) : ''}
  if (row < ROWS) {
$body
  }
  scratch[lid.x] = acc;
$reduce
  if (${t == 256 ? 'lid.x == 0u' : 'trd == 0u'} && row < ROWS) {
    y[slot * ROWS + row] = scratch[lid.x];
  }
}
''');
      s.setBuffer('wq', stack.buffer);
      s.setBuffer('x', x);
      s.setBuffer('y', y);
      s.setBuffer('idxb', _topkIdx);
      _fireRowsZ(s, (stack.rows + r - 1) ~/ r, topK);
    }

    // dp4a variant: int8 activations (xq/xsc), one hardware int8 MAC per 4
    // lanes.  THREAD-PER-ROW (256 rows/wg, each thread streams its whole
    // row, NO reduction) — this is the structure where the dot-product
    // arithmetic is the bottleneck, so cutting it with dp4a actually pays
    // off (the row-split+reduction kernel was structure-bound, not
    // arithmetic-bound, and saw no benefit).  [perSlotX] offsets into the
    // per-slot quantized intermediates (down); gate/up share _xnq.
    void fusedMatVecDp4a(QuantizedTensor stack, String tag, Buffer xq,
        Buffer xsc, Buffer y, bool perSlotX) {
      final traits = ggmlTypeTraits[stack.type]!;
      final bpe = stack.rows * stack.cols ~/ traits.blockSize * traits.typeSize;
      final body = QuantizedTensor.matVecDp4aBodyWGSL(
          threadVar: '0u',
          stride: '1u',
          xBaseWords: perSlotX ? 'slot * (COLS / 4u)' : '0u',
          xscBaseBlocks: perSlotX ? 'slot * (COLS / 32u)' : '0u');
      final s = _sh('''
// plan-expfusedq:$tag
@group(0) @binding(0) var<storage, read_write> wq: array<u32>;
@group(0) @binding(1) var<storage, read_write> xq: array<u32>;
@group(0) @binding(2) var<storage, read_write> xsc: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read_write> idxb: array<u32>;

const ROWS: u32 = ${stack.rows}u;
const COLS: u32 = ${stack.cols}u;

${QuantizedTensor.accessorsWGSL}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let slot: u32 = wid.z;
  let row: u32 = wid.x * 256u + lid.x;
  let eb: u32 = idxb[slot] * ${bpe}u;
  var acc: f32 = 0.0;
  if (row < ROWS) {
$body
  }
  if (row < ROWS) { y[slot * ROWS + row] = acc; }
}
''');
      s.setBuffer('wq', stack.buffer);
      s.setBuffer('xq', xq);
      s.setBuffer('xsc', xsc);
      s.setBuffer('y', y);
      s.setBuffer('idxb', _topkIdx);
      s.dispatchFire((stack.rows + 255) ~/ 256, 1, topK);
    }

    // Second-stage fusion (see [expFuse2]): gate+up in one z-doubled
    // dispatch writing a [2, topK, rows] plane buffer, then down with
    // silu(gate)*up computed INSIDE its shared-x staging loop.
    void fireExpGU(QuantizedTensor gkk, QuantizedTensor ukk, Buffer y) {
      final traits = ggmlTypeTraits[gkk.type]!;
      final bpe = gkk.rows * gkk.cols ~/ traits.blockSize * traits.typeSize;
      const t = 16;
      const r = 256 ~/ t;
      final accU = QuantizedTensor.accessorsWGSL
          .replaceAll('wq', 'wqu')
          .replaceAll('f16At', 'uf16At')
          .replaceAll('byteAt', 'ubyteAt')
          .replaceAll('wAt', 'uwAt');
      final bodyG = QuantizedTensor.sharedXBody(QuantizedTensor.matVecBodyWGSL(
          gkk.type,
          threadVar: 'trd',
          stride: '${t}u'));
      final bodyU =
          bodyG.replaceAll('wq[', 'wqu[').replaceAll('f16At(', 'uf16At(');
      final s = _sh('''
// plan-expgu:${gkk.rows}x${gkk.cols}
@group(0) @binding(0) var<storage, read_write> wq: array<u32>;
@group(0) @binding(1) var<storage, read_write> wqu: array<u32>;
@group(0) @binding(2) var<storage, read_write> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read_write> idxb: array<u32>;

const ROWS: u32 = ${gkk.rows}u;
const COLS: u32 = ${gkk.cols}u;
const TKR: u32 = ${topK * gkk.rows}u;

${QuantizedTensor.accessorsWGSL}
$accU
${QuantizedTensor.xsDeclWGSL(gkk.cols)}

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let slot: u32 = wid.z >> 1u;
  let gu: u32 = wid.z & 1u;
  let trd: u32 = lid.x % ${t}u;
  let row: u32 = (wid.x + wid.y * nwg.x) * ${r}u + lid.x / ${t}u;
  let eb: u32 = idxb[slot] * ${bpe}u;
  var acc: f32 = 0.0;
${QuantizedTensor.stageXsWGSL(gkk.cols)}
  if (row < ROWS) {
    if (gu == 0u) {
$bodyG
    } else {
$bodyU
    }
  }
  scratch[lid.x] = acc;
  workgroupBarrier();
  for (var s: u32 = ${t ~/ 2}u; s > 0u; s = s >> 1u) {
    if (trd < s) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s]; }
    workgroupBarrier();
  }
  if (trd == 0u && row < ROWS) {
    y[gu * TKR + slot * ROWS + row] = scratch[lid.x];
  }
}
''');
      s.setBuffer('wq', gkk.buffer);
      s.setBuffer('wqu', ukk.buffer);
      s.setBuffer('x', _xn);
      s.setBuffer('y', y);
      s.setBuffer('idxb', _topkIdx);
      s.dispatchFire((gkk.rows + r - 1) ~/ r, 1, topK * 2);
    }

    void fireExpDownSM(QuantizedTensor dkk, Buffer gu, Buffer y) {
      final traits = ggmlTypeTraits[dkk.type]!;
      final bpe = dkk.rows * dkk.cols ~/ traits.blockSize * traits.typeSize;
      const t = 16;
      const r = 256 ~/ t;
      final body = QuantizedTensor.sharedXBody(QuantizedTensor.matVecBodyWGSL(
          dkk.type,
          threadVar: 'trd',
          stride: '${t}u'));
      final s = _sh('''
// plan-expdownsm:${dkk.rows}x${dkk.cols}
@group(0) @binding(0) var<storage, read_write> wq: array<u32>;
@group(0) @binding(1) var<storage, read_write> gu: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> idxb: array<u32>;

const ROWS: u32 = ${dkk.rows}u;
const COLS: u32 = ${dkk.cols}u;
const C4: u32 = ${dkk.cols ~/ 4}u;
const TKC4: u32 = ${topK * dkk.cols ~/ 4}u;

${QuantizedTensor.accessorsWGSL}
${QuantizedTensor.xsDeclWGSL(dkk.cols)}

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let slot: u32 = wid.z;
  let trd: u32 = lid.x % ${t}u;
  let row: u32 = (wid.x + wid.y * nwg.x) * ${r}u + lid.x / ${t}u;
  let eb: u32 = idxb[slot] * ${bpe}u;
  var acc: f32 = 0.0;
  for (var i4x: u32 = lid.x; i4x < C4; i4x = i4x + 256u) {
    let gv: vec4<f32> = gu[slot * C4 + i4x];
    let uv: vec4<f32> = gu[TKC4 + slot * C4 + i4x];
    let v4x: vec4<f32> = (gv / (1.0 + exp(-gv))) * uv;
    let p4x: u32 = i4x * 4u + ((i4x * 4u) >> 5u);
    xs[p4x] = v4x.x; xs[p4x + 1u] = v4x.y;
    xs[p4x + 2u] = v4x.z; xs[p4x + 3u] = v4x.w;
  }
  workgroupBarrier();
  if (row < ROWS) {
$body
  }
  scratch[lid.x] = acc;
  workgroupBarrier();
  for (var s: u32 = ${t ~/ 2}u; s > 0u; s = s >> 1u) {
    if (trd < s) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s]; }
    workgroupBarrier();
  }
  if (trd == 0u && row < ROWS) { y[slot * ROWS + row] = scratch[lid.x]; }
}
''');
      s.setBuffer('wq', dkk.buffer);
      s.setBuffer('gu', gu);
      s.setBuffer('y', y);
      s.setBuffer('idxb', _topkIdx);
      s.dispatchFire((dkk.rows + r - 1) ~/ r, 1, topK);
    }

    final fuse2 = !dp4a &&
        expFuse2 &&
        sharedX &&
        xsRowPack &&
        _fusableType(gk.type) &&
        gk.type == uk.type && // one body serves both in plan-expgu
        _fusableType(dk.type) &&
        _sharedXOk(gk.cols) &&
        _sharedXOk(dk.cols) &&
        _expTXs(gk.cols, gk.type) == 16 &&
        _expTXs(dk.cols, dk.type) == 16 &&
        gk.rows == uk.rows &&
        gk.cols == uk.cols &&
        dk.cols == gk.rows;
    if (fuse2) {
      _guExpAll ??= _f32(2 * topK * interRows);
      fireExpGU(gk, uk, _guExpAll!);
      fireExpDownSM(dk, _guExpAll!, _downExpAll!);
    } else {
    if (dp4a) {
      fusedMatVecDp4a(gk, 'exp_gate', _xnq!, _xnsc!, _gExpAll!, false);
      fusedMatVecDp4a(uk, 'exp_up', _xnq!, _xnsc!, _uExpAll!, false);
    } else {
      fusedMatVec(gk, 'exp_gate', _xn, _gExpAll!);
      fusedMatVec(uk, 'exp_up', _xn, _uExpAll!);
    }

    // silu(gate) * up over all slots in one dispatch.
    final sm = _sh('''
// plan-expsilumul
@group(0) @binding(0) var<storage, read_write> g: array<f32>;
@group(0) @binding(1) var<storage, read_write> u: array<f32>;
@group(0) @binding(2) var<storage, read_write> prod: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i >= ${topK * interRows}u) { return; }
  let gv: f32 = g[i];
  prod[i] = (gv / (1.0 + exp(-gv))) * u[i];
}
''');
    sm.setBuffer('g', _gExpAll!);
    sm.setBuffer('u', _uExpAll!);
    sm.setBuffer('prod', _prodExpAll!);
    _fireLinear(sm, topK * interRows);

    // down: ydown[slot*dim + row] = Wdown[idxb[slot]] @ prod[slot], all slots.
    if (dp4a) {
      _fireQuantX('prod', _prodExpAll!, topK * interRows, _prodq!, _prodsc!);
      fusedMatVecDp4a(dk, 'exp_down', _prodq!, _prodsc!, _downExpAll!, true);
    } else {
      final traits = ggmlTypeTraits[dk.type]!;
      final bpe = dk.rows * dk.cols ~/ traits.blockSize * traits.typeSize;
      final xs = _sharedXOk(dk.cols);
      final t = xs && xsRowPack ? _expTXs(dk.cols, dk.type) : 256;
      final r = 256 ~/ t;
      var body = t == 256
          ? QuantizedTensor.matVecBodyWGSL(dk.type)
          : QuantizedTensor.matVecBodyWGSL(dk.type,
              threadVar: 'trd', stride: '${t}u');
      body = xs ? QuantizedTensor.sharedXBody(body) : _matVecBodyOffsetX2(body);
      final reduce = t == 256
          ? _reduceSum(256)
          : '''
  workgroupBarrier();
  for (var s: u32 = ${t ~/ 2}u; s > 0u; s = s >> 1u) {
    if (trd < s) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s]; }
    workgroupBarrier();
  }
''';
      final s = _sh('''
// plan-expdown${xs ? ':xs' : ''}${t == 256 ? '' : ':T$t'}
@group(0) @binding(0) var<storage, read_write> wq: array<u32>;
@group(0) @binding(1) var<storage, read_write> x: array<${xs ? 'vec4<f32>' : 'f32'}>;   // [topK, COLS]
@group(0) @binding(2) var<storage, read_write> y: array<f32>;   // [topK, ROWS]
@group(0) @binding(3) var<storage, read_write> idxb: array<u32>;

const ROWS: u32 = ${dk.rows}u;
const COLS: u32 = ${dk.cols}u;

${QuantizedTensor.accessorsWGSL}
${QuantizedTensor.typeNeedsScaleMinK4(dk.type) ? QuantizedTensor.scaleMinK4WGSL : ''}
${xs ? QuantizedTensor.xsDeclWGSL(dk.cols) : ''}

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let slot: u32 = wid.z;
  ${t == 256 ? '' : 'let trd: u32 = lid.x % ${t}u;'}
  let row: u32 = ${t == 256 ? 'wid.x + wid.y * nwg.x' : '(wid.x + wid.y * nwg.x) * ${r}u + lid.x / ${t}u'};
  let eb: u32 = idxb[slot] * ${bpe}u;
  ${xs ? '' : 'let xbase: u32 = slot * COLS;'}
  var acc: f32 = 0.0;
${xs ? QuantizedTensor.stageXsWGSL(dk.cols, xBase4: 'slot * ${dk.cols ~/ 4}u') : ''}
  if (row < ROWS) {
$body
  }
  scratch[lid.x] = acc;
$reduce
  if (${t == 256 ? 'lid.x == 0u' : 'trd == 0u'} && row < ROWS) {
    y[slot * ROWS + row] = scratch[lid.x];
  }
}
''');
      s.setBuffer('wq', dk.buffer);
      s.setBuffer('x', _prodExpAll!);
      s.setBuffer('y', _downExpAll!);
      s.setBuffer('idxb', _topkIdx);
      _fireRowsZ(s, (dk.rows + r - 1) ~/ r, topK);
    }
    } // end !fuse2

    // Combine: ffnOut[row] += sum_slot wsel[slot] * ydown[slot*dim + row].
    final cb = _sh('''
// plan-expcombine${directX ? (hasSh ? ':xsh' : ':x') : ''}
@group(0) @binding(0) var<storage, read_write> ydown: array<f32>; // [topK, DIM]
@group(0) @binding(1) var<storage, read_write> wsel: array<f32>;  // [topK]
@group(0) @binding(2) var<storage, read_write> ffn: array<f32>;   // [DIM] (x when directX)
${directX && hasSh ? '@group(0) @binding(3) var<storage, read_write> shout: array<f32>;' : ''}

const DIM: u32 = ${dim}u;
const K: u32 = ${topK}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let row: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (row >= DIM) { return; }
  var acc: f32 = 0.0;
  for (var s: u32 = 0u; s < K; s = s + 1u) {
    acc = acc + wsel[s] * ydown[s * DIM + row];
  }
  ${directX && hasSh ? 'ffn[row] = ffn[row] + shout[row] + acc;' : 'ffn[row] = ffn[row] + acc;'}
}
''');
    cb.setBuffer('ydown', _downExpAll!);
    cb.setBuffer('wsel', _topkW);
    // directX: this IS the residual add — the single write folds the shared
    // expert's gated output and every routed slot straight into x, so the
    // ffnOut zero pass and the res_ffn add both disappear.
    cb.setBuffer('ffn', directX ? _x : _ffnOut);
    if (directX && hasSh) cb.setBuffer('shout', _shOut!);
    _fireLinear(cb, dim);
  }

  /// matVec accumulate body variant that reads x from `xbase` (per-slot
  /// offset) instead of 0.  Wraps [QuantizedTensor.matVecBodyWGSL] output by
  /// substituting the x index base — the body indexes `x[...]`, so we alias
  /// x through a local offset by renaming.  Simpler: the down body reads the
  /// canonical body with COLS-length rows; we shift by rewriting `x[` → an
  /// offset accessor via a helper function.
  static String _matVecBodyOffsetX(int type) {
    // The shared body indexes `x[<expr>]`; make those reads hit x[xbase + …]
    // by defining a shadowing accessor.  We inject a WGSL fn xread and
    // rewrite is avoided by having the body use x directly — instead we
    // provide x already offset via a pointer is not possible in WGSL, so we
    // textually prefix each `x[` with `xbase +` through a marker.
    return QuantizedTensor.matVecBodyWGSL(type)
        .replaceAll('x[', 'x[xbase + ');
  }

  /// The same xbase rewrite applied to an ALREADY-generated body (row-split
  /// variants pass threadVar/stride into the generator first).
  static String _matVecBodyOffsetX2(String body) =>
      body.replaceAll('x[', 'x[xbase + ');

  void _fireRms(String tag, Buffer input, Buffer weight, Buffer output) {
    final s = _sh('''
// plan-rms:$tag
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const D: u32 = ${dim}u;

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  var acc: f32 = 0.0;
  for (var j: u32 = lid.x; j < D; j = j + 256u) {
    let v: f32 = input[j];
    acc = acc + v * v;
  }
  scratch[lid.x] = acc;
${_reduceSum(256)}
  let inv: f32 = inverseSqrt(scratch[0] / f32(D) + ${_f(eps)});
  for (var j: u32 = lid.x; j < D; j = j + 256u) {
    output[j] = input[j] * inv * w[j];
  }
}
''');
    s.setBuffer('input', input);
    s.setBuffer('w', weight);
    s.setBuffer('output', output);
    s.dispatchFire(1, 1, 1);
  }

  void _fireAdd(String tag, Buffer x, Buffer a) {
    final s = _sh('''
// plan-add:$tag
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> a: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i >= ${dim}u) { return; }
  x[i] = x[i] + a[i];
}
''');
    s.setBuffer('x', x);
    s.setBuffer('a', a);
    _fireLinear(s, dim);
  }

  void _fireZero(Buffer y, int n) {
    final s = _sh('''
// plan-zero
@group(0) @binding(0) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i >= ${n}u) { return; }
  y[i] = 0.0;
}
''');
    s.setBuffer('y', y);
    _fireLinear(s, n);
  }

  void _fireSiluMul(String tag, Buffer g, Buffer u, Buffer prod, int n) {
    final s = _sh('''
// plan-silumul:$tag
@group(0) @binding(0) var<storage, read_write> g: array<f32>;
@group(0) @binding(1) var<storage, read_write> u: array<f32>;
@group(0) @binding(2) var<storage, read_write> prod: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i >= ${n}u) { return; }
  let gv: f32 = g[i];
  prod[i] = (gv / (1.0 + exp(-gv))) * u[i];
}
''');
    s.setBuffer('g', g);
    s.setBuffer('u', u);
    s.setBuffer('prod', prod);
    _fireLinear(s, n);
  }

  // ---- deltanet ---------------------------------------------------------

  /// Unfused beta/alpha projections + params transform (3 dispatches).
  void _fireBetaAlphaUnfused(DeltaNetLayer d, _BlockState st) {
    _fireQmv(d.wBeta, 'dn_beta', _xn, _betaRaw!);
    _fireQmv(d.wAlpha, 'dn_alpha', _xn, _alphaRaw!);

    // g = ssmA * softplus(alpha + dtBias); beta = sigmoid(betaRaw) — on GPU
    // (kills the per-layer alpha/beta readbacks of the v1 path).
    final v = d.vHeads;
    final pk = _sh('''
// plan-dnparams
@group(0) @binding(0) var<storage, read_write> alpha: array<f32>;
@group(0) @binding(1) var<storage, read_write> betaraw: array<f32>;
@group(0) @binding(2) var<storage, read_write> consts: array<f32>; // [ssmA, dtBias]
@group(0) @binding(3) var<storage, read_write> pout: array<f32>;   // [g, beta]

const V: u32 = ${v}u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let h: u32 = gid.x;
  if (h >= V) { return; }
  let a: f32 = alpha[h] + consts[V + h];
  var sp: f32;
  if (a > 20.0) { sp = a; } else { sp = log(1.0 + exp(a)); }
  pout[h] = consts[h] * sp;
  pout[V + h] = 1.0 / (1.0 + exp(-betaraw[h]));
}
''');
    pk.setBuffer('alpha', _alphaRaw!);
    pk.setBuffer('betaraw', _betaRaw!);
    pk.setBuffer('consts', st.deltaConsts!);
    pk.setBuffer('pout', _dParams!);
    pk.dispatchFire((v + 63) ~/ 64, 1, 1);
  }

  /// FUSED beta+alpha+params: one kernel, one workgroup per output head
  /// row (2V total; V <= 64 so no grid folding needed).  Rows < V compute
  /// the beta dot and write sigmoid to pout[V+h]; rows >= V compute the
  /// alpha dot and write ssmA*softplus(dot+dtBias) to pout[h].  The alpha
  /// weight gets its OWN accessor set (renamed) since the shared
  /// accessorsWGSL text hardcodes `wq`.
  void _fireBetaAlphaParams(DeltaNetLayer d, _BlockState st) {
    final v = d.vHeads;
    final xs = _sharedXOk(d.wBeta.cols);
    // Type in the key: ssm_alpha/beta are f16 in some blocks, q8_0 in the
    // rest, at identical shapes.
    final key = 'bap/${d.wBeta.rows}x${d.wBeta.cols}/t${d.wBeta.type}'
        '${xs ? '/xs' : ''}';
    final s = _qmvShaders.putIfAbsent(key, () {
      final accA = QuantizedTensor.accessorsWGSL
          .replaceAll('wq', 'wqa')
          .replaceAll('f16At', 'af16At')
          .replaceAll('byteAt', 'abyteAt')
          .replaceAll('wAt', 'awAt');
      // The only `row` use is the row base (`row * nb` for q8_0,
      // `row * wordsPerRow` for f16).
      var bodyB = QuantizedTensor.matVecBodyWGSL(d.wBeta.type)
          .replaceAll('row *', 'rEff *');
      if (xs) bodyB = QuantizedTensor.sharedXBody(bodyB);
      final bodyA =
          bodyB.replaceAll('wq[', 'wqa[').replaceAll('f16At(', 'af16At(');
      return _sh('''
// plan-dnbap${xs ? ':xs' : ''}
@group(0) @binding(0) var<storage, read_write> wq: array<u32>;    // beta
@group(0) @binding(1) var<storage, read_write> wqa: array<u32>;   // alpha
@group(0) @binding(2) var<storage, read_write> x: array<${xs ? 'vec4<f32>' : 'f32'}>;
@group(0) @binding(3) var<storage, read_write> consts: array<f32>; // [ssmA, dtBias]
@group(0) @binding(4) var<storage, read_write> pout: array<f32>;   // [g, beta]

const COLS: u32 = ${d.wBeta.cols}u;
const V: u32 = ${v}u;

${QuantizedTensor.accessorsWGSL}
$accA
${xs ? QuantizedTensor.xsDeclWGSL(d.wBeta.cols) : ''}

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let row: u32 = wid.x;
  let isBeta: bool = row < V;
  let rEff: u32 = select(row - V, row, isBeta);
  let eb: u32 = 0u;
  var acc: f32 = 0.0;
${xs ? QuantizedTensor.stageXsWGSL(d.wBeta.cols) : ''}
  if (isBeta) {
$bodyB
  } else {
$bodyA
  }
  scratch[lid.x] = acc;
${_reduceSum(256)}
  if (lid.x == 0u) {
    let dsum: f32 = scratch[0];
    if (isBeta) {
      pout[V + rEff] = 1.0 / (1.0 + exp(-dsum));
    } else {
      let a: f32 = dsum + consts[V + rEff];
      var sp: f32;
      if (a > 20.0) { sp = a; } else { sp = log(1.0 + exp(a)); }
      pout[rEff] = consts[rEff] * sp;
    }
  }
}
''');
    });
    s.setBuffer('wq', d.wBeta.buffer);
    s.setBuffer('wqa', d.wAlpha.buffer);
    s.setBuffer('x', _xn);
    s.setBuffer('consts', st.deltaConsts!);
    s.setBuffer('pout', _dParams!);
    s.dispatchFire(2 * v, 1, 1);
  }

  void _fireDelta(PlanBlock b, _BlockState st) {
    final d = b.delta!;
    // Delta-internal timing bisect (see the skip* fields).
    final noProj = skipProj;
    final noBap = skipBap;
    final noConv = skipConv;
    final noRec = skipRec;
    final noOut = skipOut;
    // Quantize the normed input once; the two big projections (qkv, gate)
    // take the dp4a path.  beta/alpha are tiny (vHeads rows) — dp4a's
    // thread-per-row would starve them to one workgroup, so keep them f32.
    // With [chainFuse] the two projections run as ONE concatenated dispatch
    // into _qkvz; conv reads the qkv prefix in place and the fused rec
    // reads z at a baked offset.
    final catQ = !_dp4aOk(d.wqkv) && deltaFusion && _cat2Ok(d.wqkv, d.wGate);
    if (!noProj) {
      if (catQ) {
        _qkvz ??= _f32(d.wqkv.rows + d.wGate.rows);
        _fireQmvCat2(d.wqkv, d.wGate, 'dn_qkvz', _xn, _qkvz!);
      } else if (_dp4aOk(d.wqkv)) {
        _xnq ??= _u32(dim ~/ 4);
        _xnsc ??= _f32(dim ~/ 32);
        _fireQuantX('dn', _xn, dim, _xnq!, _xnsc!);
        _fireQmvDp4a(d.wqkv, 'dn_qkv', _xnq!, _xnsc!, _qkv!);
        _fireQmvDp4a(d.wGate, 'dn_z', _xnq!, _xnsc!, _z!);
      } else {
        _fireQmv(d.wqkv, 'dn_qkv', _xn, _qkv!);
        _fireQmv(d.wGate, 'dn_z', _xn, _z!);
      }
    }
    final fuseBA = deltaFusion &&
        _fusableType(d.wBeta.type) &&
        d.wBeta.type == d.wAlpha.type && // one body, renamed accessors
        d.wBeta.rows == d.vHeads &&
        d.wAlpha.rows == d.vHeads &&
        // The fused kernel bakes ONE COLS (from wBeta) for both dots.
        d.wAlpha.cols == d.wBeta.cols;
    if (!noBap) {
      if (fuseBA) {
        // beta GEMV + alpha GEMV + the params transform in ONE kernel: rows
        // 0..V-1 are beta (sigmoid -> pout[V+h]), rows V..2V-1 are alpha
        // (ssmA * softplus(dot + dtBias) -> pout[h]).  3 dispatches -> 1.
        _fireBetaAlphaParams(d, st);
      } else {
        _fireBetaAlphaUnfused(d, st);
      }
    }

    // Causal conv k=4 + SiLU; rolls raw-input history in place.
    final histLen = d.convKernel - 1;
    final conv = _sh('''
// plan-dnconv
@group(0) @binding(0) var<storage, read_write> w: array<f32>;
@group(0) @binding(1) var<storage, read_write> hist: array<f32>;
@group(0) @binding(2) var<storage, read_write> cur: array<f32>;
@group(0) @binding(3) var<storage, read_write> outv: array<f32>;

const C: u32 = ${d.convDim}u;
const K: u32 = ${d.convKernel}u;
const HIST: u32 = ${histLen}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let c: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (c >= C) { return; }
  var acc: f32 = 0.0;
  for (var t: u32 = 0u; t < HIST; t = t + 1u) {
    acc = acc + w[c * K + t] * hist[t * C + c];
  }
  let x: f32 = cur[c];
  acc = acc + w[c * K + (K - 1u)] * x;
  outv[c] = acc / (1.0 + exp(-acc));
  for (var t: u32 = 0u; t + 1u < HIST; t = t + 1u) {
    hist[t * C + c] = hist[(t + 1u) * C + c];
  }
  hist[(HIST - 1u) * C + c] = x;
}
''');
    if (!noConv) {
      conv.setBuffer('w', d.convWeight.buffer);
      conv.setBuffer('hist', d.convState.buffer);
      conv.setBuffer('cur', catQ ? _qkvz! : _qkv!);
      conv.setBuffer('outv', _convOut!);
      _fireLinear(conv, d.convDim);
    }

    // Fused split + per-head L2 norm of q and k straight out of the conv
    // output (v stays in place; the recurrence reads it at an offset).
    final hd = d.headDim;
    final l2 = _sh('''
// plan-dnl2split
@group(0) @binding(0) var<storage, read_write> conv: array<f32>;
@group(0) @binding(1) var<storage, read_write> qn: array<f32>;
@group(0) @binding(2) var<storage, read_write> kn: array<f32>;

const D: u32 = ${hd}u;
const KH: u32 = ${d.kHeads}u;
const KEYDIM: u32 = ${d.keyDim}u;

var<workgroup> scratch: array<f32, $hd>;

@compute @workgroup_size($hd)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let isQ: bool = wid.x < KH;
  let g: u32 = select(wid.x - KH, wid.x, isQ);
  let base: u32 = select(KEYDIM, 0u, isQ) + g * D;
  let v: f32 = conv[base + lid.x];
  scratch[lid.x] = v * v;
  workgroupBarrier();
  for (var s: u32 = ${hd ~/ 2}u; s > 0u; s = s >> 1u) {
    if (lid.x < s) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s]; }
    workgroupBarrier();
  }
  let inv: f32 = 1.0 / max(sqrt(scratch[0]), ${_f(d.eps)});
  let o: f32 = v * inv;
  if (isQ) { qn[g * D + lid.x] = o; } else { kn[g * D + lid.x] = o; }
}
''');
    if (!noConv) {
      l2.setBuffer('conv', _convOut!);
      l2.setBuffer('qn', _qn!);
      l2.setBuffer('kn', _kn!);
      l2.dispatchFire(2 * d.kHeads, 1, 1);
    }

    // Delta-rule recurrence (state in place); v read from the conv output.
    // With [deltaFusion] the gated norm folds into the recurrence epilogue
    // (identical vHeads x headDim grid — thread (h,j) computed exactly the
    // core element the norm needs), writing the gated output directly and
    // skipping the core round-trip + a dispatch.
    // OCCUPANCY-SPLIT recurrence (see [recSplit]).  The fused kernel runs
    // one workgroup per v-head — ~32 workgroups, so ~75% of the SMs sit idle
    // through the single biggest memory mover in the decode step.  Here TPR
    // lanes share each state row (strided i, TPR-wide tree reduction), which
    // multiplies the workgroup count by TPR and pushes the dispatch past the
    // occupancy knee.  The gated norm reduces over ALL rows of a head, so it
    // cannot ride along and goes back to its own dispatch.
    if (recSplit) {
      final tpr = _recTpr(hd, d.vHeads);
      final rpw = 256 ~/ tpr;
      final wgPerHead = (hd + rpw - 1) ~/ rpw;
      final rec = _sh('''
// plan-dnrec2:T$tpr
@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k: array<f32>;
@group(0) @binding(2) var<storage, read_write> conv: array<f32>;
@group(0) @binding(3) var<storage, read_write> state: array<f32>;
@group(0) @binding(4) var<storage, read_write> outv: array<f32>;
@group(0) @binding(5) var<storage, read_write> params: array<f32>;

const D: u32 = ${hd}u;
const KHEADS: u32 = ${d.kHeads}u;
const VHEADS: u32 = ${d.vHeads}u;
const VOFF: u32 = ${2 * d.keyDim}u;
const SCALE: f32 = ${_f(1.0 / math.sqrt(hd))};
const TPR: u32 = ${tpr}u;

var<workgroup> ks: array<f32, $hd>;
var<workgroup> qs: array<f32, $hd>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let h: u32 = wid.y;
  let kk: u32 = h % KHEADS;
  for (var i: u32 = lid.x; i < D; i = i + 256u) {
    ks[i] = k[kk * D + i];
    qs[i] = q[kk * D + i] * SCALE;
  }
  workgroupBarrier();

  let trd: u32 = lid.x % TPR;
  let grp: u32 = lid.x - trd;
  let j: u32 = wid.x * ${rpw}u + lid.x / TPR;
  let decay: f32 = exp(params[h]);
  let beta: f32 = params[VHEADS + h];
  let rowBase: u32 = (h * D + j) * D;

  var sk: f32 = 0.0;
  if (j < D) {
    for (var i: u32 = trd; i < D; i = i + TPR) {
      let s: f32 = state[rowBase + i] * decay;
      state[rowBase + i] = s;
      sk = sk + s * ks[i];
    }
  }
  red[lid.x] = sk;
  workgroupBarrier();
  for (var s: u32 = TPR >> 1u; s > 0u; s = s >> 1u) {
    if (trd < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
    workgroupBarrier();
  }
  let skT: f32 = red[grp];
  workgroupBarrier();

  var dlt: f32 = 0.0;
  if (j < D) { dlt = (conv[VOFF + h * D + j] - skT) * beta; }

  var o: f32 = 0.0;
  if (j < D) {
    for (var i: u32 = trd; i < D; i = i + TPR) {
      let s: f32 = state[rowBase + i] + ks[i] * dlt;
      state[rowBase + i] = s;
      o = o + s * qs[i];
    }
  }
  red[lid.x] = o;
  workgroupBarrier();
  for (var s: u32 = TPR >> 1u; s > 0u; s = s >> 1u) {
    if (trd < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
    workgroupBarrier();
  }
  if (trd == 0u && j < D) { outv[h * D + j] = red[lid.x]; }
}
''');
      if (!noRec) {
        rec.setBuffer('q', _qn!);
        rec.setBuffer('k', _kn!);
        rec.setBuffer('conv', _convOut!);
        rec.setBuffer('state', d.ssmState.buffer);
        rec.setBuffer('outv', _core!);
        rec.setBuffer('params', _dParams!);
        rec.dispatchFire(wgPerHead, d.vHeads, 1);
      }
      _fireGatedNorm(d, hd, catQ);
      if (!noOut) _fireOutProj(d.wOut, 'dn_out', _dGated!);
      return;
    }

    if (deltaFusion) {
      final rec = _sh('''
// plan-dnrecg
@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k: array<f32>;
@group(0) @binding(2) var<storage, read_write> conv: array<f32>;
@group(0) @binding(3) var<storage, read_write> state: array<f32>;
@group(0) @binding(4) var<storage, read_write> params: array<f32>;
@group(0) @binding(5) var<storage, read_write> z: array<f32>;
@group(0) @binding(6) var<storage, read_write> nw: array<f32>;
@group(0) @binding(7) var<storage, read_write> outb: array<f32>;

const D: u32 = ${hd}u;
const KHEADS: u32 = ${d.kHeads}u;
const VHEADS: u32 = ${d.vHeads}u;
const VOFF: u32 = ${2 * d.keyDim}u;
const SCALE: f32 = ${_f(1.0 / math.sqrt(hd))};

var<workgroup> ks: array<f32, $hd>;
var<workgroup> qs: array<f32, $hd>;
var<workgroup> scratch: array<f32, $hd>;

@compute @workgroup_size($hd)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let h: u32 = wid.x;
  let j: u32 = lid.x;
  let kk: u32 = h % KHEADS;

  ks[j] = k[kk * D + j];
  qs[j] = q[kk * D + j] * SCALE;
  workgroupBarrier();

  let decay: f32 = exp(params[h]);
  let beta: f32 = params[VHEADS + h];
  let rowBase: u32 = (h * D + j) * D;

  var sk: f32 = 0.0;
  for (var i: u32 = 0u; i < D; i = i + 1u) {
    let s: f32 = state[rowBase + i] * decay;
    state[rowBase + i] = s;
    sk = sk + s * ks[i];
  }

  let dlt: f32 = (conv[VOFF + h * D + j] - sk) * beta;

  var o: f32 = 0.0;
  for (var i: u32 = 0u; i < D; i = i + 1u) {
    let s: f32 = state[rowBase + i] + ks[i] * dlt;
    state[rowBase + i] = s;
    o = o + s * qs[i];
  }

  // Gated norm epilogue: rmsnorm(o over the head, ssmNorm) * silu(z).
  scratch[j] = o * o;
  workgroupBarrier();
  for (var s: u32 = ${hd ~/ 2}u; s > 0u; s = s >> 1u) {
    if (j < s) { scratch[j] = scratch[j] + scratch[j + s]; }
    workgroupBarrier();
  }
  let inv: f32 = inverseSqrt(scratch[0] / f32(D) + ${_f(d.eps)});
  let zv: f32 = z[${catQ ? d.wqkv.rows : 0}u + h * D + j];
  outb[h * D + j] = o * inv * nw[j] * (zv / (1.0 + exp(-zv)));
}
''');
      if (!noRec) {
        rec.setBuffer('q', _qn!);
        rec.setBuffer('k', _kn!);
        rec.setBuffer('conv', _convOut!);
        rec.setBuffer('state', d.ssmState.buffer);
        rec.setBuffer('params', _dParams!);
        rec.setBuffer('z', catQ ? _qkvz! : _z!);
        rec.setBuffer('nw', d.ssmNorm.buffer);
        rec.setBuffer('outb', _dGated!);
        rec.dispatchFire(d.vHeads, 1, 1);
      }
      if (!noOut) _fireOutProj(d.wOut, 'dn_out', _dGated!);
      return;
    }

    final rec = _sh('''
// plan-dnrec
@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k: array<f32>;
@group(0) @binding(2) var<storage, read_write> conv: array<f32>;
@group(0) @binding(3) var<storage, read_write> state: array<f32>;
@group(0) @binding(4) var<storage, read_write> outv: array<f32>;
@group(0) @binding(5) var<storage, read_write> params: array<f32>;

const D: u32 = ${hd}u;
const KHEADS: u32 = ${d.kHeads}u;
const VHEADS: u32 = ${d.vHeads}u;
const VOFF: u32 = ${2 * d.keyDim}u;
const SCALE: f32 = ${_f(1.0 / math.sqrt(hd))};

var<workgroup> ks: array<f32, $hd>;
var<workgroup> qs: array<f32, $hd>;

@compute @workgroup_size($hd)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let h: u32 = wid.x;
  let j: u32 = lid.x;
  let kk: u32 = h % KHEADS;

  ks[j] = k[kk * D + j];
  qs[j] = q[kk * D + j] * SCALE;
  workgroupBarrier();

  let decay: f32 = exp(params[h]);
  let beta: f32 = params[VHEADS + h];
  let rowBase: u32 = (h * D + j) * D;

  var sk: f32 = 0.0;
  for (var i: u32 = 0u; i < D; i = i + 1u) {
    let s: f32 = state[rowBase + i] * decay;
    state[rowBase + i] = s;
    sk = sk + s * ks[i];
  }

  let dlt: f32 = (conv[VOFF + h * D + j] - sk) * beta;

  var o: f32 = 0.0;
  for (var i: u32 = 0u; i < D; i = i + 1u) {
    let s: f32 = state[rowBase + i] + ks[i] * dlt;
    state[rowBase + i] = s;
    o = o + s * qs[i];
  }
  outv[h * D + j] = o;
}
''');
    rec.setBuffer('q', _qn!);
    rec.setBuffer('k', _kn!);
    rec.setBuffer('conv', _convOut!);
    rec.setBuffer('state', d.ssmState.buffer);
    rec.setBuffer('outv', _core!);
    rec.setBuffer('params', _dParams!);
    if (!skipRec) {
      rec.dispatchFire(d.vHeads, 1, 1);
    }

    _fireGatedNorm(d, hd, catQ);
    _fireOutProj(d.wOut, 'dn_out', _dGated!);
  }

  /// Gated norm epilogue: rmsnorm(core over the head, ssmNorm) * silu(z),
  /// one workgroup per v-head.  [catQ] says this block's z lives in the
  /// chain-fused projection buffer, after the qkv rows.
  void _fireGatedNorm(DeltaNetLayer d, int hd, bool catQ) {
    final zIdx = catQ ? '${d.wqkv.rows}u + h * D + lid.x' : 'h * D + lid.x';
    final gn = _sh('''
// plan-dngnorm${catQ ? ':o' : ''}
@group(0) @binding(0) var<storage, read_write> core: array<f32>;
@group(0) @binding(1) var<storage, read_write> z: array<f32>;
@group(0) @binding(2) var<storage, read_write> nw: array<f32>;
@group(0) @binding(3) var<storage, read_write> outb: array<f32>;

const D: u32 = ${hd}u;

var<workgroup> scratch: array<f32, $hd>;

@compute @workgroup_size($hd)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let h: u32 = wid.x;
  let v: f32 = core[h * D + lid.x];
  scratch[lid.x] = v * v;
  workgroupBarrier();
  for (var s: u32 = ${hd ~/ 2}u; s > 0u; s = s >> 1u) {
    if (lid.x < s) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s]; }
    workgroupBarrier();
  }
  let inv: f32 = inverseSqrt(scratch[0] / f32(D) + ${_f(d.eps)});
  let zv: f32 = z[$zIdx];
  outb[h * D + lid.x] = v * inv * nw[lid.x] * (zv / (1.0 + exp(-zv)));
}
''');
    gn.setBuffer('core', _core!);
    gn.setBuffer('z', catQ ? _qkvz! : _z!);
    gn.setBuffer('nw', d.ssmNorm.buffer);
    gn.setBuffer('outb', _dGated!);
    gn.dispatchFire(d.vHeads, 1, 1);
  }

  // ---- attention --------------------------------------------------------

  void _fireAttn(PlanBlock b, _BlockState st, int pos) {
    final a = b.attn!;
    if (_dp4aOk(a.wq)) {
      _xnq ??= _u32(dim ~/ 4);
      _xnsc ??= _f32(dim ~/ 32);
      _fireQuantX('attn', _xn, dim, _xnq!, _xnsc!);
      _fireQmvDp4a(a.wq, 'attn_wq', _xnq!, _xnsc!, _qFull!);
      _fireQmvDp4a(a.wk, 'attn_wk', _xnq!, _xnsc!, _kRaw!);
      _fireQmvDp4a(a.wv, 'attn_wv', _xnq!, _xnsc!, _vRaw!);
    } else {
      _fireQmv(a.wq, 'attn_wq', _xn, _qFull!);
      _fireQmv(a.wk, 'attn_wk', _xn, _kRaw!);
      _fireQmv(a.wv, 'attn_wv', _xn, _vRaw!);
    }

    final hd = a.headDim;
    // Fused per-head QK RMS-norm + partial NEOX rope + KV-cache append.
    // Position comes from the pos buffer, so nothing bakes seqLen.
    final prep = _sh('''
// plan-attnprep
@group(0) @binding(0) var<storage, read_write> qfull: array<f32>;
@group(0) @binding(1) var<storage, read_write> kraw: array<f32>;
@group(0) @binding(2) var<storage, read_write> vraw: array<f32>;
@group(0) @binding(3) var<storage, read_write> qnw: array<f32>;
@group(0) @binding(4) var<storage, read_write> knw: array<f32>;
@group(0) @binding(5) var<storage, read_write> posb: array<u32>;
@group(0) @binding(6) var<storage, read_write> qout: array<f32>;
@group(0) @binding(7) var<storage, read_write> gateout: array<f32>;
@group(0) @binding(8) var<storage, read_write> kcache: array<f32>;
@group(0) @binding(9) var<storage, read_write> vcache: array<f32>;

const HEADS: u32 = ${a.heads}u;
const D: u32 = ${hd}u;
const ROT: u32 = ${a.ropeDims}u;
const HALF: u32 = ${a.ropeDims ~/ 2}u;
const KVDIM: u32 = ${a.kvDim}u;
const THETA: f32 = ${_f(a.ropeThetaBase)};

var<workgroup> rowv: array<f32, $hd>;
var<workgroup> scratch: array<f32, $hd>;

@compute @workgroup_size($hd)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let j: u32 = lid.x;
  let isQ: bool = wid.x < HEADS;
  let g: u32 = select(wid.x - HEADS, wid.x, isQ);
  var v: f32;
  if (isQ) { v = qfull[wid.x * 2u * D + j]; } else { v = kraw[g * D + j]; }
  scratch[j] = v * v;
  workgroupBarrier();
  for (var s: u32 = ${hd ~/ 2}u; s > 0u; s = s >> 1u) {
    if (lid.x < s) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s]; }
    workgroupBarrier();
  }
  let inv: f32 = inverseSqrt(scratch[0] / f32(D) + ${_f(a.eps)});
  rowv[j] = v * inv * select(knw[j], qnw[j], isQ);
  workgroupBarrier();

  let p: u32 = posb[0];
  let fpos: f32 = f32(p);
  var o: f32;
  if (j < HALF) {
    let ang: f32 = fpos * pow(THETA, -f32(2u * j) / f32(ROT));
    o = rowv[j] * cos(ang) - rowv[j + HALF] * sin(ang);
  } else if (j < ROT) {
    let i: u32 = j - HALF;
    let ang: f32 = fpos * pow(THETA, -f32(2u * i) / f32(ROT));
    o = rowv[i] * sin(ang) + rowv[j] * cos(ang);
  } else {
    o = rowv[j];
  }
  if (isQ) {
    qout[wid.x * D + j] = o;
    gateout[wid.x * D + j] = qfull[wid.x * 2u * D + D + j];
  } else {
    kcache[p * KVDIM + g * D + j] = o;
    vcache[p * KVDIM + g * D + j] = vraw[g * D + j];
  }
}
''');
    prep.setBuffer('qfull', _qFull!);
    prep.setBuffer('kraw', _kRaw!);
    prep.setBuffer('vraw', _vRaw!);
    prep.setBuffer('qnw', a.qNorm.buffer);
    prep.setBuffer('knw', a.kNorm.buffer);
    prep.setBuffer('posb', _pos);
    prep.setBuffer('qout', _q!);
    prep.setBuffer('gateout', _gate!);
    prep.setBuffer('kcache', st.kCache!);
    prep.setBuffer('vcache', st.vCache!);
    prep.dispatchFire(a.heads + a.kvHeads, 1, 1);

    // Scores: one workgroup per (t, head); seqLen enters as the DISPATCH
    // size (pos is CPU-known), never the shader source.
    final group = a.heads ~/ a.kvHeads;
    final sc = _sh('''
// plan-attnscores
@group(0) @binding(0) var<storage, read_write> qb: array<f32>;
@group(0) @binding(1) var<storage, read_write> kcache: array<f32>;
@group(0) @binding(2) var<storage, read_write> sc: array<f32>;

const D: u32 = ${hd}u;
const KVDIM: u32 = ${a.kvDim}u;
const GROUP: u32 = ${group}u;
const MAXSEQ: u32 = ${maxSeq}u;
const SCALE: f32 = ${_f(1.0 / math.sqrt(hd))};

var<workgroup> scratch: array<f32, $hd>;

@compute @workgroup_size($hd)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let t: u32 = wid.x;
  let h: u32 = wid.y;
  let kv: u32 = h / GROUP;
  scratch[lid.x] = qb[h * D + lid.x] * kcache[t * KVDIM + kv * D + lid.x];
${_reduceSum(hd)}
  if (lid.x == 0u) { sc[h * MAXSEQ + t] = scratch[0] * SCALE; }
}
''');
    sc.setBuffer('qb', _q!);
    sc.setBuffer('kcache', st.kCache!);
    sc.setBuffer('sc', _scores!);
    sc.dispatchFire(pos + 1, a.heads, 1);

    // Softmax over the live prefix + weighted-V + sigmoid output gate, one
    // workgroup per q-head; seqLen read from the pos buffer.
    final sv = _sh('''
// plan-attnsv
@group(0) @binding(0) var<storage, read_write> sc: array<f32>;
@group(0) @binding(1) var<storage, read_write> vcache: array<f32>;
@group(0) @binding(2) var<storage, read_write> gateb: array<f32>;
@group(0) @binding(3) var<storage, read_write> posb: array<u32>;
@group(0) @binding(4) var<storage, read_write> outb: array<f32>;

const D: u32 = ${hd}u;
const KVDIM: u32 = ${a.kvDim}u;
const GROUP: u32 = ${group}u;
const MAXSEQ: u32 = ${maxSeq}u;

var<workgroup> scratch: array<f32, $hd>;

@compute @workgroup_size($hd)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let h: u32 = wid.x;
  let n: u32 = posb[0] + 1u;
  var m: f32 = -1.0e30;
  for (var t: u32 = lid.x; t < n; t = t + ${hd}u) {
    m = max(m, sc[h * MAXSEQ + t]);
  }
  scratch[lid.x] = m;
  workgroupBarrier();
  for (var s: u32 = ${hd ~/ 2}u; s > 0u; s = s >> 1u) {
    if (lid.x < s) { scratch[lid.x] = max(scratch[lid.x], scratch[lid.x + s]); }
    workgroupBarrier();
  }
  let mx: f32 = scratch[0];
  workgroupBarrier();
  var sm: f32 = 0.0;
  for (var t: u32 = lid.x; t < n; t = t + ${hd}u) {
    let e: f32 = exp(sc[h * MAXSEQ + t] - mx);
    sc[h * MAXSEQ + t] = e;
    sm = sm + e;
  }
  storageBarrier();
  scratch[lid.x] = sm;
${_reduceSum(hd)}
  let total: f32 = scratch[0];
  let kv: u32 = h / GROUP;
  var acc: f32 = 0.0;
  for (var t: u32 = 0u; t < n; t = t + 1u) {
    acc = acc + sc[h * MAXSEQ + t] * vcache[t * KVDIM + kv * D + lid.x];
  }
  let gv: f32 = gateb[h * D + lid.x];
  outb[h * D + lid.x] = (acc / total) * (1.0 / (1.0 + exp(-gv)));
}
''');
    sv.setBuffer('sc', _scores!);
    sv.setBuffer('vcache', st.vCache!);
    sv.setBuffer('gateb', _gate!);
    sv.setBuffer('posb', _pos);
    sv.setBuffer('outb', _gated!);
    sv.dispatchFire(a.heads, 1, 1);

    _fireOutProj(a.wo, 'attn_wo', _gated!);
  }

  // ---- MoE --------------------------------------------------------------

  /// Returns true when the MoE already accumulated its result into x (the
  /// [residFuse] combine), so the caller must NOT fire a residual add.
  Future<bool> _moe(PlanBlock b) async {
    final m = b.moe;
    final experts = m.router.shape[0];
    if (experts > 256) {
      throw Exception('top-k kernel supports <= 256 experts, got $experts');
    }

    // Path selection up front: the router folds the shared expert's gate
    // dot as one extra row (killing a whole single-workgroup dispatch per
    // block), and the combine writes straight into x when it is this
    // block's only FFN writer.
    final shDp4a = _dp4aExperts && (m.gateShexp?.type == GgmlType.q8_0);
    final expDp4a = _dp4aExperts &&
        m.isResident &&
        m.stackGate?.type == GgmlType.q8_0 &&
        m.stackUp?.type == GgmlType.q8_0 &&
        m.stackDown?.type == GgmlType.q8_0;
    final noShexpFlag = skipShexp;
    final noExpFlag = skipExp;
    final hasSh = m.gateShexp != null && !noShexpFlag;
    final gsh0 = m.gateShexp, ush0 = m.upShexp, dsh0 = m.downShexp;
    final shFuse = hasSh &&
        !shDp4a &&
        ush0 != null &&
        dsh0 != null &&
        _cat2Ok(gsh0!, ush0) &&
        gsh0.rows == ush0.rows &&
        _fusableType(dsh0.type) &&
        dsh0.cols == gsh0.rows &&
        _sharedXOk(dsh0.cols) &&
        _expTXs(dsh0.cols, dsh0.type) == 16;
    // The NO_EXP ablation keeps the old accumulate-into-ffnOut path so its
    // timing stays comparable with the control arm.
    final directX =
        residFuse && m.isResident && !noExpFlag && (!hasSh || shFuse);
    final foldGate = hasSh && chainFuse && m.sharedGate != null;

    // Router matVec (f32 weights); row ROWS is the shared-expert gate.
    final rt = _sh('''
// plan-router${foldGate ? ':g' : ''}
@group(0) @binding(0) var<storage, read_write> wf: array<f32>;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
${foldGate ? '''@group(0) @binding(3) var<storage, read_write> wsh: array<f32>;
@group(0) @binding(4) var<storage, read_write> outs: array<f32>;''' : ''}

const ROWS: u32 = ${experts}u;
const COLS: u32 = ${dim}u;

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let row: u32 = wid.x + wid.y * nwg.x;
  var acc: f32 = 0.0;
  if (row < ROWS) {
    for (var c: u32 = lid.x; c < COLS; c = c + 256u) {
      acc = acc + wf[row * COLS + c] * x[c];
    }
  }${foldGate ? ''' else if (row == ROWS) {
    for (var c: u32 = lid.x; c < COLS; c = c + 256u) {
      acc = acc + wsh[c] * x[c];
    }
  }''' : ''}
  scratch[lid.x] = acc;
${_reduceSum(256)}
  if (lid.x == 0u && row < ROWS) { y[row] = scratch[0]; }
${foldGate ? '  if (lid.x == 0u && row == ROWS) { outs[0] = 1.0 / (1.0 + exp(-scratch[0])); }' : ''}
}
''');
    rt.setBuffer('wf', m.router.buffer);
    rt.setBuffer('x', _xn);
    rt.setBuffer('y', _routLogits);
    if (foldGate) {
      rt.setBuffer('wsh', m.sharedGate!.buffer);
      rt.setBuffer('outs', _shScalar!);
    }
    if (!skipRoute) _fireRows(rt, foldGate ? experts + 1 : experts);

    // Softmax + top-k + renormalized weights, all on GPU: only the selected
    // ids come back to the CPU (to bind streamed expert weights).
    final tk = _sh('''
// plan-topk
@group(0) @binding(0) var<storage, read_write> lg: array<f32>;
@group(0) @binding(1) var<storage, read_write> idxout: array<u32>;
@group(0) @binding(2) var<storage, read_write> wout: array<f32>;

const E: u32 = ${experts}u;
const K: u32 = ${topK}u;

var<workgroup> p: array<f32, $experts>;
var<workgroup> sval: array<f32, 256>;
var<workgroup> sidx: array<u32, 256>;
var<workgroup> sel: array<u32, $topK>;
var<workgroup> sw: array<f32, $topK>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let e: u32 = lid.x;
  var v: f32 = -1.0e30;
  if (e < E) { v = lg[e]; }
  sval[lid.x] = v;
  workgroupBarrier();
  for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
    if (lid.x < s) { sval[lid.x] = max(sval[lid.x], sval[lid.x + s]); }
    workgroupBarrier();
  }
  let mx: f32 = sval[0];
  workgroupBarrier();
  if (e < E) { p[e] = exp(lg[e] - mx); }
  workgroupBarrier();

  // RANK-SELECT: one pass instead of K sequential max-extractions.  rank(e)
  // counts the experts that outrank e — strictly greater, or equal with a
  // lower index, which is exactly the tie-break of the extraction loop this
  // replaces (its reduction swapped only on STRICTLY greater, so ties kept
  // the lower index).  Identical winners in identical slots; ~64 fewer
  // workgroup barriers in a kernel that runs 40x per token on ONE
  // workgroup (the whole GPU idles behind it).
  if (e < E) {
    let pe: f32 = p[e];
    var rank: u32 = 0u;
    for (var f: u32 = 0u; f < E; f = f + 1u) {
      let pf: f32 = p[f];
      if (pf > pe || (pf == pe && f < e)) { rank = rank + 1u; }
    }
    if (rank < K) {
      sel[rank] = e;
      sw[rank] = pe;
    }
  }
  workgroupBarrier();

  if (lid.x == 0u) {
    var total: f32 = 0.0;
    for (var s: u32 = 0u; s < K; s = s + 1u) { total = total + sw[s]; }
    for (var s: u32 = 0u; s < K; s = s + 1u) {
      idxout[s] = sel[s];
      wout[s] = sw[s] / total;
    }
  }
}
''');
    tk.setBuffer('lg', _routLogits);
    tk.setBuffer('idxout', _topkIdx);
    tk.setBuffer('wout', _topkW);
    if (!skipRoute) tk.dispatchFire(1, 1, 1);

    // Quantize the normed hidden state to int8 ONCE for the dp4a path;
    // shared expert gate/up + all routed experts reuse it.
    if (shDp4a || expDp4a) {
      _xnq ??= _u32(dim ~/ 4);
      _xnsc ??= _f32(dim ~/ 32);
      _fireQuantX('xn', _xn, dim, _xnq!, _xnsc!);
    }

    if (!directX) _fireZero(_ffnOut, dim);

    // Shared expert fires BEFORE the routing readback so the GPU overlaps it
    // with the CPU-side expert fetching.  Its sigmoid(dot) gate rides along
    // in the router dispatch unless [foldGate] is off.
    if (hasSh) {
      if (!foldGate) {
      final sg = _sh('''
// plan-shscalar
@group(0) @binding(0) var<storage, read_write> w: array<f32>;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> outs: array<f32>;

const N: u32 = ${dim}u;

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  var acc: f32 = 0.0;
  for (var i: u32 = lid.x; i < N; i = i + 256u) {
    acc = acc + w[i] * x[i];
  }
  scratch[lid.x] = acc;
${_reduceSum(256)}
  if (lid.x == 0u) { outs[0] = 1.0 / (1.0 + exp(-scratch[0])); }
}
''');
      sg.setBuffer('w', m.sharedGate!.buffer);
      sg.setBuffer('x', _xn);
      sg.setBuffer('outs', _shScalar!);
      sg.dispatchFire(1, 1, 1);
      } // end !foldGate (otherwise the router computed the gate)

      final gsh = m.gateShexp!, ush = m.upShexp!, dsh = m.downShexp!;
      if (shFuse) {
        // gate+up concatenated in one dispatch, then down with silu·mul
        // computed inside its staging loop, accumulating via the sigmoid
        // gate scalar — 4 dispatches -> 2.
        _guSh ??= _f32(2 * gsh.rows);
        _fireQmvCat2(gsh, ush, 'sh_gu', _xn, _guSh!);
        const t = 16;
        const r = 256 ~/ t;
        final body = QuantizedTensor.sharedXBody(
            QuantizedTensor.matVecBodyWGSL(dsh.type,
                threadVar: 'trd', stride: '${t}u'));
        final s = _sh('''
// plan-shdownsm:${dsh.rows}x${dsh.cols}
@group(0) @binding(0) var<storage, read_write> wq: array<u32>;
@group(0) @binding(1) var<storage, read_write> gu: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> wsel: array<f32>;

const ROWS: u32 = ${dsh.rows}u;
const COLS: u32 = ${dsh.cols}u;
const C4: u32 = ${dsh.cols ~/ 4}u;

${QuantizedTensor.accessorsWGSL}
${QuantizedTensor.xsDeclWGSL(dsh.cols)}

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let trd: u32 = lid.x % ${t}u;
  let row: u32 = (wid.x + wid.y * nwg.x) * ${r}u + lid.x / ${t}u;
  let eb: u32 = 0u;
  var acc: f32 = 0.0;
  for (var i4x: u32 = lid.x; i4x < C4; i4x = i4x + 256u) {
    let gv: vec4<f32> = gu[i4x];
    let uv: vec4<f32> = gu[C4 + i4x];
    let v4x: vec4<f32> = (gv / (1.0 + exp(-gv))) * uv;
    let p4x: u32 = i4x * 4u + ((i4x * 4u) >> 5u);
    xs[p4x] = v4x.x; xs[p4x + 1u] = v4x.y;
    xs[p4x + 2u] = v4x.z; xs[p4x + 3u] = v4x.w;
  }
  workgroupBarrier();
  if (row < ROWS) {
$body
  }
  scratch[lid.x] = acc;
  workgroupBarrier();
  for (var s: u32 = ${t ~/ 2}u; s > 0u; s = s >> 1u) {
    if (trd < s) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s]; }
    workgroupBarrier();
  }
  if (trd == 0u && row < ROWS) {
    ${directX ? 'y[row] = wsel[0u] * scratch[lid.x];' : 'y[row] = y[row] + wsel[0u] * scratch[lid.x];'}
  }
}
''');
        if (directX) _shOut ??= _f32(dim);
        s.setBuffer('wq', dsh.buffer);
        s.setBuffer('gu', _guSh!);
        s.setBuffer('y', directX ? _shOut! : _ffnOut);
        s.setBuffer('wsel', _shScalar!);
        s.dispatchFire((dsh.rows + r - 1) ~/ r, 1, 1);
      } else {
        if (shDp4a) {
          _fireQmvDp4a(gsh, 'sh_gate', _xnq!, _xnsc!, _gSh!);
          _fireQmvDp4a(ush, 'sh_up', _xnq!, _xnsc!, _uSh!);
        } else {
          _fireQmv(gsh, 'sh_gate', _xn, _gSh!);
          _fireQmv(ush, 'sh_up', _xn, _uSh!);
        }
        _fireSiluMul('sh', _gSh!, _uSh!, _prodSh!, gsh.rows);
        _fireQmvAccum(dsh, 'sh_down', _prodSh!, _ffnOut, _shScalar!, 0);
      }
    }

    // Resident stacks: expert ids never leave the GPU — all top-K slots run
    // in 5 fused dispatches (vs 4/slot).  Sync-free block.
    if (m.isResident) {
      if (!noExpFlag) {
        _fireExpertsFused(m, directX: directX, hasSh: hasSh);
      }
      return directX;
    }

    // Streamed path — THE per-block sync: which experts won.  This readback
    // also flushes every fired dispatch before it, making the slot rebinds
    // below safe.
    final sw = Stopwatch()..start();
    final idx = Uint32List(topK);
    await _topkIdx.read(idx, topK, dataType: BufferDataType.uint32);
    syncMicros += sw.elapsedMicroseconds;

    // Start ALL fetches up front so disk reads pipeline with each other and
    // with the GPU; bind + fire each slot as its three parts arrive.
    final parts = [
      for (int s = 0; s < topK; s++)
        (
          g: m.fetchExpert('g', idx[s]),
          u: m.fetchExpert('u', idx[s]),
          d: m.fetchExpert('d', idx[s]),
        ),
    ];
    for (int s = 0; s < topK; s++) {
      final g = await parts[s].g;
      final u = await parts[s].u;
      final dn = await parts[s].d;
      _fireQmv(g, 'exp_gate$s', _xn, _gExp ??= _f32(g.rows));
      _fireQmv(u, 'exp_up$s', _xn, _uExp ??= _f32(u.rows));
      _fireSiluMul('exp$s', _gExp!, _uExp!, _prodExp ??= _f32(g.rows), g.rows);
      _fireQmvAccum(dn, 'exp_down$s', _prodExp!, _ffnOut, _topkW, s);
    }
    return false; // streamed path accumulates into _ffnOut
  }

  // ---- token step -------------------------------------------------------

  /// Writes the hidden state (embedding row, or the upstream device's
  /// hidden state in a multi-GPU pipeline) into this plan's x buffer.
  Future<void> writeX(Float32List x) async {
    if (x.length != dim) {
      throw Exception('x has ${x.length} elements, dim is $dim');
    }
    await _x.write(x, dim);
  }

  /// Reads the hidden state back (the cross-device hop, ~dim*4 bytes).
  /// Also acts as a full FIFO flush for this device.
  Future<Float32List> readX() async {
    final out = Float32List(dim);
    final sw = Stopwatch()..start();
    await _x.read(out, dim);
    syncMicros += sw.elapsedMicroseconds;
    return out;
  }

  /// Position already in the GPU-side [_pos] buffer, or -2 when unknown
  /// (start of life, after prefill's direct writes).
  int _lastGpuPos = -2;

  /// Runs this plan's blocks for the token at [pos].  Fully fired except the
  /// streamed-MoE routing readbacks.
  Future<void> runBlocks(int pos) async {
    if (pos >= maxSeq) {
      throw Exception('position $pos exceeds KV capacity $maxSeq');
    }
    if (pos == _lastGpuPos + 1) {
      // Sequential decode: bump the GPU-side counter with a fired 1-thread
      // kernel instead of a queue write — the write executes at SUBMISSION
      // time, which forces a per-token batch flush to keep it ordered.
      final s = _sh('''
// plan-posbump
@group(0) @binding(0) var<storage, read_write> pos: array<u32>;
@compute @workgroup_size(1)
fn main() { pos[0] = pos[0] + 1u; }
''');
      s.setBuffer('pos', _pos);
      s.dispatchFire(1, 1, 1);
    } else {
      final posData = Uint32List(4);
      posData[0] = pos;
      await _pos.write(posData, 4, dataType: BufferDataType.uint32);
    }
    _lastGpuPos = pos;

    // Decode ablation flags (timing bisect only — output is garbage):
    // measure a category's cost by diffing warm tok/s with vs without, on
    // the UN-profiled full run (drain-profiling inflates fixed slots).
    final noAttn = skipAttn;
    final noMoe = skipMoe;
    final noDelta = skipDelta;
    final noAttnOnly = skipAttnOnly;

    // Decode drain-profile (compile-time): buckets inflate ~4x at fixed
    // positions (drains serialize the pipeline) — use the PROPORTIONS only.
    const prof = bool.fromEnvironment('GPU_ML_DEC_PROFILE');
    Future<void> mark(String k) async {
      if (!prof) return;
      final probe = Float32List(1);
      final sw = Stopwatch()..start();
      await _x.read(probe, 1);
      _profBuckets[k] = (_profBuckets[k] ?? 0) + sw.elapsedMicroseconds;
    }

    for (int i = 0; i < blocks.length; i++) {
      final b = blocks[i];
      final st = _blockState[i];
      if (!skipNorms) _fireRms('attn_norm', _x, b.attnNorm.buffer, _xn);
      if (!noAttn) {
        if (b.delta != null) {
          if (!noDelta) {
            _fireDelta(b, st);
            if (prof) await mark('delta');
          }
        } else {
          if (!noAttnOnly) {
            _fireAttn(b, st, pos);
            if (prof) await mark('attn');
          }
        }
        // With residFuse the output projection already accumulated into x.
        if (!residFuse) _fireAdd('res_attn', _x, _aOut);
      }
      if (!skipNorms) _fireRms('post_norm', _x, b.postNorm.buffer, _xn);
      if (!noMoe) {
        final wroteX = await _moe(b);
        if (prof) await mark('moe');
        if (!wroteX) _fireAdd('res_ffn', _x, _ffnOut);
      }
    }
    if (prof) {
      await mark('norms+adds');
      final parts = _profBuckets.entries
          .map((e) => '${e.key}=${(e.value / 1000).toStringAsFixed(1)}ms')
          .join(' ');
      // ignore: avoid_print
      print('decode-profile pos=$pos: $parts');
      _profBuckets.clear();
    }
  }

  /// Fires final norm + lm_head into the logits buffer (no readback).
  void _fireHead() {
    final logitsBuf = _logits!;
    _fireRms('out_norm', _x, outputNorm!.buffer, _xn);
    if (_dp4aHead) {
      // 248k rows -> thread-per-row dp4a with full occupancy; the head is
      // the single biggest decode matVec (~540 MB q8_0 read per token).
      _xnq ??= _u32(dim ~/ 4);
      _xnsc ??= _f32(dim ~/ 32);
      _fireQuantX('head', _xn, dim, _xnq!, _xnsc!);
      _fireQmvDp4a(lmHead!, 'lm_head', _xnq!, _xnsc!, logitsBuf);
    } else {
      _fireQmv(lmHead!, 'lm_head', _xn, logitsBuf);
    }
  }

  /// Final norm + head + logits readback.  Only valid on the plan that owns
  /// [lmHead] (the last device).
  Future<Float32List> readLogits() async {
    _fireHead();
    final logits = Float32List(vocab);
    final sw = Stopwatch()..start();
    await _logits!.read(logits, vocab);
    syncMicros += sw.elapsedMicroseconds;
    return logits;
  }

  Buffer? _amaxV, _amaxI, _amaxTok;
  static const _amaxWgs = 128;

  /// Fires a two-stage greedy argmax over the logits into [_amaxTok].
  /// Tie-break matches the CPU scan: the LOWEST index among equal maxima.
  void _fireArgmax() {
    _amaxV ??= _f32(_amaxWgs);
    _amaxI ??= _u32(_amaxWgs);
    _amaxTok ??= _u32(4);
    final s1 = _sh('''
// plan-argmax1
@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> pv: array<f32>;
@group(0) @binding(2) var<storage, read_write> pi: array<u32>;

const VOCAB: u32 = ${vocab}u;

var<workgroup> sv: array<f32, 256>;
var<workgroup> si: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  var bv: f32 = -3.0e38;
  var bi: u32 = 0u;
  for (var i: u32 = wid.x * 256u + lid.x; i < VOCAB; i = i + ${_amaxWgs * 256}u) {
    let v: f32 = logits[i];
    if (v > bv) { bv = v; bi = i; }
  }
  sv[lid.x] = bv;
  si[lid.x] = bi;
  workgroupBarrier();
  for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
    if (lid.x < s) {
      let o: u32 = lid.x + s;
      if (sv[o] > sv[lid.x] || (sv[o] == sv[lid.x] && si[o] < si[lid.x])) {
        sv[lid.x] = sv[o];
        si[lid.x] = si[o];
      }
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    pv[wid.x] = sv[0];
    pi[wid.x] = si[0];
  }
}
''');
    s1.setBuffer('logits', _logits!);
    s1.setBuffer('pv', _amaxV!);
    s1.setBuffer('pi', _amaxI!);
    s1.dispatchFire(_amaxWgs, 1, 1);

    final s2 = _sh('''
// plan-argmax2
@group(0) @binding(0) var<storage, read_write> pv: array<f32>;
@group(0) @binding(1) var<storage, read_write> pi: array<u32>;
@group(0) @binding(2) var<storage, read_write> tok: array<u32>;

var<workgroup> sv: array<f32, 256>;
var<workgroup> si: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  if (lid.x < ${_amaxWgs}u) {
    sv[lid.x] = pv[lid.x];
    si[lid.x] = pi[lid.x];
  } else {
    sv[lid.x] = -3.0e38;
    si[lid.x] = 0u;
  }
  workgroupBarrier();
  for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
    if (lid.x < s) {
      let o: u32 = lid.x + s;
      if (sv[o] > sv[lid.x] || (sv[o] == sv[lid.x] && si[o] < si[lid.x])) {
        sv[lid.x] = sv[o];
        si[lid.x] = si[o];
      }
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) { tok[0] = si[0]; }
}
''');
    s2.setBuffer('pv', _amaxV!);
    s2.setBuffer('pi', _amaxI!);
    s2.setBuffer('tok', _amaxTok!);
    s2.dispatchFire(1, 1, 1);
  }

  /// Final norm + head + GPU argmax + 4-BYTE readback: the greedy token id
  /// without pulling ~600 KB of logits over PCIe or running a ~250k-element
  /// CPU scan per token.  [_amaxTok] keeps the id GPU-resident so the next
  /// token's embed gather can chain from it without a CPU write.
  Future<int> readArgmaxToken() async {
    _fireHead();
    _fireArgmax();
    final out = Uint32List(4);
    final sw = Stopwatch()..start();
    await _amaxTok!.read(out, 1, dataType: BufferDataType.uint32);
    syncMicros += sw.elapsedMicroseconds;
    return out[0];
  }

  /// One single-device decode step: [embedRow] is the CPU-dequantized
  /// embedding for the token, [pos] its position.  Returns the logits.
  Future<Float32List> forward(Float32List embedRow, int pos) async {
    await writeX(embedRow);
    await runBlocks(pos);
    return readLogits();
  }

  // ---- BATCHED PREFILL ---------------------------------------------------
  // Processes a chunk of up to [maxPrefillChunk] prompt tokens in one fired
  // pass per op instead of one sequential forward per token — decode is
  // dispatch-launch bound, so batching the prompt amortizes ~T× of that
  // overhead.  Resident-stack blocks only (the runner falls back to
  // sequential forwards otherwise).  The DeltaNet recurrence is inherently
  // sequential over tokens; it runs its whole T-step loop INSIDE one kernel.

  static const int maxPrefillChunk = 256;

  bool get supportsBatchedPrefill =>
      blocks.every((b) => b.moe.isResident);

  Buffer? _pT; // [4] u32: [0] = chunk length T
  Buffer? _px, _pxn, _paOut, _pffn;
  Buffer? _pqFull, _pkRaw, _pvRaw, _pq, _pgate, _pgated;
  Buffer? _pqkv, _pz, _pbetaRaw, _palphaRaw, _pdParams, _pconvOut, _pqn, _pkn,
      _pcore, _pdGated;
  Buffer? _pRoutLogits, _pTopkIdx, _pTopkW, _pShScalar;
  Buffer? _pgSh, _puSh, _pprodSh, _pshDown;
  Buffer? _pgExp, _puExp, _pprodExp, _pdownExp;
  // Expert grouping: assignments (t,slot) sorted by expert so each expert's
  // weights are read once per token-tile instead of once per assignment.
  Buffer? _pExpOff, _pSorted, _pWl, _pWlc, _pTok;
  // Prefill dp4a: int8-quantized activations for the grouped expert kernels
  // (normed hidden rows + silu(g)*u intermediates), per-32-block scales.
  Buffer? _pXnq, _pXnsc, _pProdq, _pProdsc;
  final Map<String, int> _profBuckets = {};

  void _ensurePrefillBuffers() {
    if (_px != null) return;
    const T = maxPrefillChunk;
    _pT = _u32(4);
    _px = _f32(T * dim);
    _pxn = _f32(T * dim);
    _paOut = _f32(T * dim);
    _pffn = _f32(T * dim);
    final attn = blocks.map((b) => b.attn).whereType<AttentionLayer>();
    if (attn.isNotEmpty) {
      final a = attn.first;
      _pqFull = _f32(T * a.wq.rows);
      _pkRaw = _f32(T * a.kvDim);
      _pvRaw = _f32(T * a.kvDim);
      _pq = _f32(T * a.qDim);
      _pgate = _f32(T * a.qDim);
      _pgated = _f32(T * a.qDim);
    }
    final delta = blocks.map((b) => b.delta).whereType<DeltaNetLayer>();
    if (delta.isNotEmpty) {
      final d = delta.first;
      _pqkv = _f32(T * d.convDim);
      _pz = _f32(T * d.valueDim);
      _pbetaRaw = _f32(T * d.vHeads);
      _palphaRaw = _f32(T * d.vHeads);
      _pdParams = _f32(T * 2 * d.vHeads);
      _pconvOut = _f32(T * d.convDim);
      _pqn = _f32(T * d.keyDim);
      _pkn = _f32(T * d.keyDim);
      _pcore = _f32(T * d.valueDim);
      _pdGated = _f32(T * d.valueDim);
    }
    final experts = blocks.first.moe.router.shape[0];
    _pRoutLogits = _f32(T * experts);
    _pTopkIdx = _u32(T * topK);
    _pTopkW = _f32(T * topK);
    _pShScalar = _f32(T);
    final sh = blocks.map((b) => b.moe.gateShexp).whereType<QuantizedTensor>();
    if (sh.isNotEmpty) {
      _pgSh = _f32(T * sh.first.rows);
      _puSh = _f32(T * sh.first.rows);
      _pprodSh = _f32(T * sh.first.rows);
      _pshDown = _f32(T * dim);
    }
    final st = blocks.firstWhere((b) => b.moe.isResident).moe;
    _pgExp = _f32(T * topK * st.stackGate!.rows);
    _puExp = _f32(T * topK * st.stackGate!.rows);
    _pprodExp = _f32(T * topK * st.stackGate!.rows);
    _pdownExp = _f32(T * topK * dim);
    _pExpOff = _u32(experts + 1);
    _pSorted = _u32(T * topK);
    _pWl = _u32(T * topK ~/ _grpTB + experts + 1);
    _pWlc = _u32(4);
    _pTok = _u32(T);
    // Match the grouped-dp4a use condition (review finding): all three
    // stacks q8_0 and a top-k kernel-compatible expert count.
    if (st.stackGate?.type == GgmlType.q8_0 &&
        st.stackUp?.type == GgmlType.q8_0 &&
        st.stackDown?.type == GgmlType.q8_0 &&
        experts <= 256) {
      _pXnq = _u32(T * dim ~/ 4);
      _pXnsc = _f32(T * dim ~/ 32);
      _pProdq = _u32(T * topK * st.stackGate!.rows ~/ 4);
      _pProdsc = _f32(T * topK * st.stackGate!.rows ~/ 32);
    }

    // f16 weight scratch for the dense-GEMM prefill path: sized to the
    // largest dense projection (elements / 2 packed u32 words).
    var maxDense = 0;
    void see(QuantizedTensor? w) {
      if (w != null && w.rows * w.cols > maxDense) maxDense = w.rows * w.cols;
    }

    for (final b in blocks) {
      final a = b.attn;
      if (a != null) {
        see(a.wq);
        see(a.wk);
        see(a.wv);
        see(a.wo);
      }
      final d = b.delta;
      if (d != null) {
        see(d.wqkv);
        see(d.wGate);
        see(d.wBeta);
        see(d.wAlpha);
        see(d.wOut);
      }
      see(b.moe.gateShexp);
      see(b.moe.upShexp);
      see(b.moe.downShexp);
    }
    _pwF16 = _u32(maxDense ~/ 2);
  }

  Buffer? _pwF16;

  /// Batched dense projection over T token rows: y[t*ROWS+row] = W @ x[t].
  /// For real chunks the weight is dequantized ONCE into the shared f16
  /// scratch and multiplied with a 16x16-tiled GEMM, so weight bytes are
  /// read once per 16 tokens instead of once per token — dense prefill is
  /// bandwidth-bound on exactly that traffic.  Tiny chunks fall back to the
  /// per-token matVec.
  ///
  /// With [xq]/[xsc] (the caller's pre-quantized int8 activations for [x])
  /// and a q8_0 weight, the dense INT8 GEMM runs instead: no f16 scratch
  /// pass at all, hardware int8 MACs (prefillDp4a).
  void _fireQmvB(QuantizedTensor w, String tag, Buffer x, Buffer y, int T,
      {Buffer? xq, Buffer? xsc}) {
    if (prefillDp4a &&
        xq != null &&
        xsc != null &&
        T >= 16 &&
        w.cols % 32 == 0 &&
        w.type == GgmlType.q8_0) {
      _fireGemmQ(w, tag, xq, xsc, y, T);
      return;
    }
    if (T >= 16 && w.cols % 16 == 0 && w.type == GgmlType.f16) {
      // f16 weights are ALREADY packed f16 pairs — GEMM straight off them.
      _fireGemmB(w.rows, w.cols, x, y, T, w16: w.buffer);
      return;
    }
    if (T >= 16 && w.cols % 16 == 0 && w.type == GgmlType.q8_0) {
      _fireDequantF16Q8(w);
      _fireGemmB(w.rows, w.cols, x, y, T);
      return;
    }
    // K-quants and tiny chunks: per-token matVec fallback.
    _fireQmvBNarrow(w, tag, x, y, T);
  }

  /// Dequantizes a Q8_0 weight into the shared packed-f16 scratch, one
  /// thread per 34-byte block using whole-word funnel loads — the same
  /// proven pattern as the vectorized matVec body.  (A per-byte accessor
  /// variant produced wrong lane-3 reads under FXC — do not reintroduce.)
  void _fireDequantF16Q8(QuantizedTensor w) {
    final n = w.rows * w.cols;
    final s = _qmvShaders.putIfAbsent('deq16q8/$n', () => _sh('''
// plan-deqf16q8:$n
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
'''));
    s.setBuffer('wq', w.buffer);
    s.setBuffer('w16', _pwF16!);
    _fireLinear(s, n ~/ 32);
  }

  /// Tiled GEMM: y[t, r] = sum_c x[t, c] * f16w[r, c], 16x16x16 tiles.
  void _fireGemmB(int rows, int cols, Buffer x, Buffer y, int T,
      {Buffer? w16}) {
    final s = _qmvShaders.putIfAbsent('gemm/${rows}x$cols', () => _sh('''
// plan-gemmB:${rows}x$cols
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
'''));
    s.setBuffer('w16', w16 ?? _pwF16!);
    s.setBuffer('x', x);
    s.setBuffer('y', y);
    s.setBuffer('pT', _pT!);
    s.dispatchFire((rows + 15) ~/ 16, (T + 15) ~/ 16, 1);
  }

  /// Per-token fallback for tiny chunks: y[t*ROWS+row] = W @ x[t].
  void _fireQmvBNarrow(
      QuantizedTensor w, String tag, Buffer x, Buffer y, int T) {
    final key = 'B/$tag/${w.rows}x${w.cols}/t${w.type}';
    final s = _qmvShaders.putIfAbsent(key, () => _sh('''
// plan-qmvB:$tag
@group(0) @binding(0) var<storage, read_write> wq: array<u32>;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

const ROWS: u32 = ${w.rows}u;
const COLS: u32 = ${w.cols}u;

${QuantizedTensor.accessorsWGSL}
${QuantizedTensor.typeNeedsScaleMinK4(w.type) ? QuantizedTensor.scaleMinK4WGSL : ''}

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let t: u32 = wid.z;
  let row: u32 = wid.x + wid.y * nwg.x;
  let eb: u32 = 0u;
  let xbase: u32 = t * COLS;
  var acc: f32 = 0.0;
  if (row < ROWS) {
${_matVecBodyOffsetX(w.type)}
  }
  scratch[lid.x] = acc;
${_reduceSum(256)}
  if (lid.x == 0u && row < ROWS) { y[t * ROWS + row] = scratch[0]; }
}
'''));
    s.setBuffer('wq', w.buffer);
    s.setBuffer('x', x);
    s.setBuffer('y', y);
    _fireRowsZ(s, w.rows, T);
  }

  /// Batched RMS norm: out[t] = rms(in[t]) * w, one workgroup per token.
  void _fireRmsB(String tag, Buffer input, Buffer weight, Buffer output,
      int T) {
    final s = _sh('''
// plan-rmsB:$tag
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const D: u32 = ${dim}u;

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let base: u32 = wid.x * D;
  var acc: f32 = 0.0;
  for (var j: u32 = lid.x; j < D; j = j + 256u) {
    let v: f32 = input[base + j];
    acc = acc + v * v;
  }
  scratch[lid.x] = acc;
${_reduceSum(256)}
  let inv: f32 = inverseSqrt(scratch[0] / f32(D) + ${_f(eps)});
  for (var j: u32 = lid.x; j < D; j = j + 256u) {
    output[base + j] = input[base + j] * inv * w[j];
  }
}
''');
    s.setBuffer('input', input);
    s.setBuffer('w', weight);
    s.setBuffer('output', output);
    s.dispatchFire(T, 1, 1);
  }

  void _fireAddB(String tag, Buffer x, Buffer a, int n) {
    final s = _sh('''
// plan-addB:$tag
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> a: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i >= ${maxPrefillChunk * dim}u) { return; }
  x[i] = x[i] + a[i];
}
''');
    s.setBuffer('x', x);
    s.setBuffer('a', a);
    _fireLinear(s, n);
  }

  void _fireAttnB(PlanBlock b, _BlockState st, int T) {
    final a = b.attn!;
    final qd = prefillDp4a && _pXnq != null && T >= 16;
    if (qd) _fireQuantXB('pxn', _pxn!, maxPrefillChunk * dim, _pXnq!, _pXnsc!);
    final xq = qd ? _pXnq : null, xsc = qd ? _pXnsc : null;
    _fireQmvB(a.wq, 'attn_wq', _pxn!, _pqFull!, T, xq: xq, xsc: xsc);
    _fireQmvB(a.wk, 'attn_wk', _pxn!, _pkRaw!, T, xq: xq, xsc: xsc);
    _fireQmvB(a.wv, 'attn_wv', _pxn!, _pvRaw!, T, xq: xq, xsc: xsc);

    final hd = a.headDim;
    // Per-token QK norm + rope + KV append; z = token.
    final prep = _sh('''
// plan-attnprepB
@group(0) @binding(0) var<storage, read_write> qfull: array<f32>;
@group(0) @binding(1) var<storage, read_write> kraw: array<f32>;
@group(0) @binding(2) var<storage, read_write> vraw: array<f32>;
@group(0) @binding(3) var<storage, read_write> qnw: array<f32>;
@group(0) @binding(4) var<storage, read_write> knw: array<f32>;
@group(0) @binding(5) var<storage, read_write> posb: array<u32>;
@group(0) @binding(6) var<storage, read_write> qout: array<f32>;
@group(0) @binding(7) var<storage, read_write> gateout: array<f32>;
@group(0) @binding(8) var<storage, read_write> kcache: array<f32>;
@group(0) @binding(9) var<storage, read_write> vcache: array<f32>;

const HEADS: u32 = ${a.heads}u;
const D: u32 = ${hd}u;
const ROT: u32 = ${a.ropeDims}u;
const HALF: u32 = ${a.ropeDims ~/ 2}u;
const KVDIM: u32 = ${a.kvDim}u;
const QROW: u32 = ${a.wq.rows}u;
const QDIM: u32 = ${a.qDim}u;
const THETA: f32 = ${_f(a.ropeThetaBase)};

var<workgroup> rowv: array<f32, $hd>;
var<workgroup> scratch: array<f32, $hd>;

@compute @workgroup_size($hd)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let t: u32 = wid.z;
  let j: u32 = lid.x;
  let isQ: bool = wid.x < HEADS;
  let g: u32 = select(wid.x - HEADS, wid.x, isQ);
  var v: f32;
  if (isQ) { v = qfull[t * QROW + wid.x * 2u * D + j]; }
  else { v = kraw[t * KVDIM + g * D + j]; }
  scratch[j] = v * v;
  workgroupBarrier();
  for (var s: u32 = ${hd ~/ 2}u; s > 0u; s = s >> 1u) {
    if (lid.x < s) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s]; }
    workgroupBarrier();
  }
  let inv: f32 = inverseSqrt(scratch[0] / f32(D) + ${_f(a.eps)});
  rowv[j] = v * inv * select(knw[j], qnw[j], isQ);
  workgroupBarrier();

  let p: u32 = posb[0] + t;
  let fpos: f32 = f32(p);
  var o: f32;
  if (j < HALF) {
    let ang: f32 = fpos * pow(THETA, -f32(2u * j) / f32(ROT));
    o = rowv[j] * cos(ang) - rowv[j + HALF] * sin(ang);
  } else if (j < ROT) {
    let i: u32 = j - HALF;
    let ang: f32 = fpos * pow(THETA, -f32(2u * i) / f32(ROT));
    o = rowv[i] * sin(ang) + rowv[j] * cos(ang);
  } else {
    o = rowv[j];
  }
  if (isQ) {
    qout[t * QDIM + wid.x * D + j] = o;
    gateout[t * QDIM + wid.x * D + j] = qfull[t * QROW + wid.x * 2u * D + D + j];
  } else {
    kcache[p * KVDIM + g * D + j] = o;
    vcache[p * KVDIM + g * D + j] = vraw[t * KVDIM + g * D + j];
  }
}
''');
    prep.setBuffer('qfull', _pqFull!);
    prep.setBuffer('kraw', _pkRaw!);
    prep.setBuffer('vraw', _pvRaw!);
    prep.setBuffer('qnw', a.qNorm.buffer);
    prep.setBuffer('knw', a.kNorm.buffer);
    prep.setBuffer('posb', _pos);
    prep.setBuffer('qout', _pq!);
    prep.setBuffer('gateout', _pgate!);
    prep.setBuffer('kcache', st.kCache!);
    prep.setBuffer('vcache', st.vCache!);
    prep.dispatchFire(a.heads + a.kvHeads, 1, T);

    // Causal attention per (token, head): scores live in workgroup shared
    // memory (maxSeq * 4 bytes — needs the 32 KB workgroup-storage limit we
    // request from the adapter).
    final group = a.heads ~/ a.kvHeads;
    final sv = _sh('''
// plan-attnsvB
@group(0) @binding(0) var<storage, read_write> qb: array<f32>;
@group(0) @binding(1) var<storage, read_write> kcache: array<f32>;
@group(0) @binding(2) var<storage, read_write> vcache: array<f32>;
@group(0) @binding(3) var<storage, read_write> gateb: array<f32>;
@group(0) @binding(4) var<storage, read_write> posb: array<u32>;
@group(0) @binding(5) var<storage, read_write> outb: array<f32>;

const D: u32 = ${hd}u;
const KVDIM: u32 = ${a.kvDim}u;
const GROUP: u32 = ${group}u;
const QDIM: u32 = ${a.qDim}u;

var<workgroup> sc: array<f32, $maxSeq>;
var<workgroup> scratch: array<f32, $hd>;

@compute @workgroup_size($hd)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let t: u32 = wid.x;
  let h: u32 = wid.y;
  let n: u32 = posb[0] + t + 1u;
  let kv: u32 = h / GROUP;
  let qbase: u32 = t * QDIM + h * D;

  // Scores for this (token, head): each thread owns strided keys.
  for (var key: u32 = lid.x; key < n; key = key + ${hd}u) {
    var dot: f32 = 0.0;
    for (var j: u32 = 0u; j < D; j = j + 1u) {
      dot = dot + qb[qbase + j] * kcache[key * KVDIM + kv * D + j];
    }
    sc[key] = dot * ${_f(1.0 / math.sqrt(hd))};
  }
  workgroupBarrier();

  var m: f32 = -1.0e30;
  for (var key: u32 = lid.x; key < n; key = key + ${hd}u) {
    m = max(m, sc[key]);
  }
  scratch[lid.x] = m;
  workgroupBarrier();
  for (var s: u32 = ${hd ~/ 2}u; s > 0u; s = s >> 1u) {
    if (lid.x < s) { scratch[lid.x] = max(scratch[lid.x], scratch[lid.x + s]); }
    workgroupBarrier();
  }
  let mx: f32 = scratch[0];
  workgroupBarrier();
  var sm: f32 = 0.0;
  for (var key: u32 = lid.x; key < n; key = key + ${hd}u) {
    let e: f32 = exp(sc[key] - mx);
    sc[key] = e;
    sm = sm + e;
  }
  scratch[lid.x] = sm;
${_reduceSum(hd)}
  let total: f32 = scratch[0];
  workgroupBarrier();
  var acc: f32 = 0.0;
  for (var key: u32 = 0u; key < n; key = key + 1u) {
    acc = acc + sc[key] * vcache[key * KVDIM + kv * D + lid.x];
  }
  let gv: f32 = gateb[t * QDIM + h * D + lid.x];
  outb[t * QDIM + h * D + lid.x] = (acc / total) * (1.0 / (1.0 + exp(-gv)));
}
''');
    sv.setBuffer('qb', _pq!);
    sv.setBuffer('kcache', st.kCache!);
    sv.setBuffer('vcache', st.vCache!);
    sv.setBuffer('gateb', _pgate!);
    sv.setBuffer('posb', _pos);
    sv.setBuffer('outb', _pgated!);
    sv.dispatchFire(T, a.heads, 1);

    _fireQmvB(a.wo, 'attn_wo', _pgated!, _paOut!, T);
  }

  /// Rows per recurrence workgroup: the largest power-of-two divisor of [d]
  /// that keeps the f32 state slice within the 32KB groupshared budget and
  /// the workgroup at <= 64 threads.
  static int _recRowsPerWg(int d) {
    var rh = 64;
    while (rh > 1 && (d % rh != 0 || d * rh * 4 > 32768)) {
      rh ~/= 2;
    }
    return rh;
  }

  void _fireDeltaB(PlanBlock b, _BlockState st, int T) {
    final d = b.delta!;
    // Quantize the normed rows once; every q8_0 projection reading _pxn
    // takes the dense int8 GEMM (no f16 dequant pass).  T >= 16 matches
    // _fireQmvB's dp4a gate — below it the quantize has no consumer
    // (review finding: dead full-capacity dispatches on tiny tail chunks).
    final qd = prefillDp4a && _pXnq != null && T >= 16;
    if (qd) _fireQuantXB('pxn', _pxn!, maxPrefillChunk * dim, _pXnq!, _pXnsc!);
    final xq = qd ? _pXnq : null, xsc = qd ? _pXnsc : null;
    _fireQmvB(d.wqkv, 'dn_qkv', _pxn!, _pqkv!, T, xq: xq, xsc: xsc);
    _fireQmvB(d.wGate, 'dn_z', _pxn!, _pz!, T, xq: xq, xsc: xsc);
    _fireQmvB(d.wBeta, 'dn_beta', _pxn!, _pbetaRaw!, T, xq: xq, xsc: xsc);
    _fireQmvB(d.wAlpha, 'dn_alpha', _pxn!, _palphaRaw!, T, xq: xq, xsc: xsc);

    final v = d.vHeads;
    final pk = _sh('''
// plan-dnparamsB
@group(0) @binding(0) var<storage, read_write> alpha: array<f32>;
@group(0) @binding(1) var<storage, read_write> betaraw: array<f32>;
@group(0) @binding(2) var<storage, read_write> consts: array<f32>;
@group(0) @binding(3) var<storage, read_write> pout: array<f32>;

const V: u32 = ${v}u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 64u);
  if (i >= ${maxPrefillChunk * v}u) { return; }
  let t: u32 = i / V;
  let h: u32 = i % V;
  let a: f32 = alpha[i] + consts[V + h];
  var sp: f32;
  if (a > 20.0) { sp = a; } else { sp = log(1.0 + exp(a)); }
  pout[t * 2u * V + h] = consts[h] * sp;
  pout[t * 2u * V + V + h] = 1.0 / (1.0 + exp(-betaraw[i]));
}
''');
    pk.setBuffer('alpha', _palphaRaw!);
    pk.setBuffer('betaraw', _pbetaRaw!);
    pk.setBuffer('consts', st.deltaConsts!);
    pk.setBuffer('pout', _pdParams!);
    final threads = T * v;
    final wg = (threads + 63) ~/ 64;
    pk.dispatchFire(wg == 0 ? 1 : wg, 1, 1);

    // Causal conv over the chunk (parallel across (t, c)) + SiLU; the roll
    // of the history buffer happens in a second kernel.  Requires
    // T >= convKernel - 1 (the runner falls back to sequential otherwise).
    final histLen = d.convKernel - 1;
    final conv = _sh('''
// plan-dnconvB
@group(0) @binding(0) var<storage, read_write> w: array<f32>;
@group(0) @binding(1) var<storage, read_write> hist: array<f32>;
@group(0) @binding(2) var<storage, read_write> xin: array<f32>;
@group(0) @binding(3) var<storage, read_write> outv: array<f32>;
@group(0) @binding(4) var<storage, read_write> pT: array<u32>;

const C: u32 = ${d.convDim}u;
const K: u32 = ${d.convKernel}u;
const HIST: u32 = ${histLen}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i >= pT[0] * C) { return; }
  let t: u32 = i / C;
  let c: u32 = i % C;
  var acc: f32 = 0.0;
  for (var k: u32 = 0u; k < K; k = k + 1u) {
    // source token index t - (K-1) + k relative to the chunk start
    let src: i32 = i32(t) - i32(K - 1u) + i32(k);
    var xv: f32;
    if (src >= 0) { xv = xin[u32(src) * C + c]; }
    else { xv = hist[u32(i32(HIST) + src) * C + c]; }
    acc = acc + w[c * K + k] * xv;
  }
  outv[i] = acc / (1.0 + exp(-acc));
}
''');
    conv.setBuffer('w', d.convWeight.buffer);
    conv.setBuffer('hist', d.convState.buffer);
    conv.setBuffer('xin', _pqkv!);
    conv.setBuffer('outv', _pconvOut!);
    conv.setBuffer('pT', _pT!);
    _fireLinear(conv, T * d.convDim);

    // Roll history: hist[j] = raw input at chunk position T - HIST + j.
    // Safe without a temp because T >= HIST (reads only hit xin).
    final roll = _sh('''
// plan-dnhistB
@group(0) @binding(0) var<storage, read_write> hist: array<f32>;
@group(0) @binding(1) var<storage, read_write> xin: array<f32>;
@group(0) @binding(2) var<storage, read_write> pT: array<u32>;

const C: u32 = ${d.convDim}u;
const HIST: u32 = ${histLen}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i >= ${histLen}u * C) { return; }
  let j: u32 = i / C;
  let c: u32 = i % C;
  hist[i] = xin[(pT[0] - HIST + j) * C + c];
}
''');
    roll.setBuffer('hist', d.convState.buffer);
    roll.setBuffer('xin', _pqkv!);
    roll.setBuffer('pT', _pT!);
    _fireLinear(roll, histLen * d.convDim);

    final hd = d.headDim;
    final l2 = _sh('''
// plan-dnl2splitB
@group(0) @binding(0) var<storage, read_write> conv: array<f32>;
@group(0) @binding(1) var<storage, read_write> qn: array<f32>;
@group(0) @binding(2) var<storage, read_write> kn: array<f32>;

const D: u32 = ${hd}u;
const KH: u32 = ${d.kHeads}u;
const KEYDIM: u32 = ${d.keyDim}u;
const CONVDIM: u32 = ${d.convDim}u;

var<workgroup> scratch: array<f32, $hd>;

@compute @workgroup_size($hd)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let t: u32 = wid.z;
  let isQ: bool = wid.x < KH;
  let g: u32 = select(wid.x - KH, wid.x, isQ);
  let base: u32 = t * CONVDIM + select(KEYDIM, 0u, isQ) + g * D;
  let v: f32 = conv[base + lid.x];
  scratch[lid.x] = v * v;
  workgroupBarrier();
  for (var s: u32 = ${hd ~/ 2}u; s > 0u; s = s >> 1u) {
    if (lid.x < s) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s]; }
    workgroupBarrier();
  }
  let inv: f32 = 1.0 / max(sqrt(scratch[0]), ${_f(d.eps)});
  let o: f32 = v * inv;
  let dst: u32 = t * KEYDIM + g * D + lid.x;
  if (isQ) { qn[dst] = o; } else { kn[dst] = o; }
}
''');
    l2.setBuffer('conv', _pconvOut!);
    l2.setBuffer('qn', _pqn!);
    l2.setBuffer('kn', _pkn!);
    l2.dispatchFire(2 * d.kHeads, 1, T);

    // Rows per recurrence workgroup: a divisor of D fitting the 32KB
    // groupshared budget (D*RH*4 <= 32768), capped at 64 threads.
    // (Helper defined below _fireDeltaB.)
    //
    // Sequential T-step recurrence inside ONE kernel (state carries across
    // tokens); T is baked (few chunk sizes, one-time compiles).  Each
    // workgroup owns an RH-row slice of one head's DxD state IN SHARED
    // MEMORY (column-major — row-major strides land every thread on the
    // same bank), loaded once before the loop and stored once after.  Rows
    // are thread-private and ks/qs are broadcast L1 reads, so the T-loop
    // has NO barriers — the storage-resident version round-tripped the
    // whole state through L2 twice per step and was 56% of prefill.
    final rh = _recRowsPerWg(hd);
    final rec = _sh('''
// plan-dnrecB:T=$T
@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k: array<f32>;
@group(0) @binding(2) var<storage, read_write> conv: array<f32>;
@group(0) @binding(3) var<storage, read_write> state: array<f32>;
@group(0) @binding(4) var<storage, read_write> outv: array<f32>;
@group(0) @binding(5) var<storage, read_write> params: array<f32>;

const D: u32 = ${hd}u;
const KHEADS: u32 = ${d.kHeads}u;
const VHEADS: u32 = ${d.vHeads}u;
const KEYDIM: u32 = ${d.keyDim}u;
const CONVDIM: u32 = ${d.convDim}u;
const VOFF: u32 = ${2 * d.keyDim}u;
const T: u32 = ${T}u;
const RH: u32 = ${rh}u;
const SCALE: f32 = ${_f(1.0 / math.sqrt(hd))};

var<workgroup> st: array<f32, ${rh * hd}>;

@compute @workgroup_size($rh)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let h: u32 = wid.x;
  let jj: u32 = lid.x;
  let j: u32 = wid.y * RH + jj;
  let kk: u32 = h % KHEADS;
  let segBase: u32 = (h * D + wid.y * RH) * D;

  // Cooperative coalesced load into column-major shared.
  for (var idx: u32 = lid.x; idx < RH * D; idx = idx + RH) {
    st[(idx % D) * RH + idx / D] = state[segBase + idx];
  }
  workgroupBarrier();

  for (var t: u32 = 0u; t < T; t = t + 1u) {
    let decay: f32 = exp(params[t * 2u * VHEADS + h]);
    let beta: f32 = params[t * 2u * VHEADS + VHEADS + h];
    let kb: u32 = t * KEYDIM + kk * D;

    var sk: f32 = 0.0;
    for (var i: u32 = 0u; i < D; i = i + 1u) {
      let s: f32 = st[i * RH + jj] * decay;
      st[i * RH + jj] = s;
      sk = sk + s * k[kb + i];
    }
    let dlt: f32 = (conv[t * CONVDIM + VOFF + h * D + j] - sk) * beta;
    var o: f32 = 0.0;
    for (var i: u32 = 0u; i < D; i = i + 1u) {
      let s: f32 = st[i * RH + jj] + k[kb + i] * dlt;
      st[i * RH + jj] = s;
      o = o + s * (q[kb + i] * SCALE);
    }
    outv[t * VHEADS * D + h * D + j] = o;
  }

  workgroupBarrier();
  for (var idx: u32 = lid.x; idx < RH * D; idx = idx + RH) {
    state[segBase + idx] = st[(idx % D) * RH + idx / D];
  }
}
''');
    rec.setBuffer('q', _pqn!);
    rec.setBuffer('k', _pkn!);
    rec.setBuffer('conv', _pconvOut!);
    rec.setBuffer('state', d.ssmState.buffer);
    rec.setBuffer('outv', _pcore!);
    rec.setBuffer('params', _pdParams!);
    rec.dispatchFire(d.vHeads, hd ~/ rh, 1);

    final gn = _sh('''
// plan-dngnormB
@group(0) @binding(0) var<storage, read_write> core: array<f32>;
@group(0) @binding(1) var<storage, read_write> z: array<f32>;
@group(0) @binding(2) var<storage, read_write> nw: array<f32>;
@group(0) @binding(3) var<storage, read_write> outb: array<f32>;

const D: u32 = ${hd}u;
const VDIM: u32 = ${d.valueDim}u;

var<workgroup> scratch: array<f32, $hd>;

@compute @workgroup_size($hd)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let t: u32 = wid.z;
  let h: u32 = wid.x;
  let base: u32 = t * VDIM + h * D;
  let v: f32 = core[base + lid.x];
  scratch[lid.x] = v * v;
  workgroupBarrier();
  for (var s: u32 = ${hd ~/ 2}u; s > 0u; s = s >> 1u) {
    if (lid.x < s) { scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s]; }
    workgroupBarrier();
  }
  let inv: f32 = inverseSqrt(scratch[0] / f32(D) + ${_f(d.eps)});
  let zv: f32 = z[base + lid.x];
  outb[base + lid.x] = v * inv * nw[lid.x] * (zv / (1.0 + exp(-zv)));
}
''');
    gn.setBuffer('core', _pcore!);
    gn.setBuffer('z', _pz!);
    gn.setBuffer('nw', d.ssmNorm.buffer);
    gn.setBuffer('outb', _pdGated!);
    gn.dispatchFire(d.vHeads, 1, T);

    _fireQmvB(d.wOut, 'dn_out', _pdGated!, _paOut!, T);
  }

  // ---- expert grouping (prefill) -----------------------------------------
  // The fused-expert kernels read each expert's weights once per (token,
  // slot) assignment — T*K full-weight reads per chunk.  Grouping sorts the
  // assignments by expert on the GPU and processes them in tiles of _grpTB
  // tokens, so each expert's weights are read ceil(cnt/TB) times instead of
  // cnt times (~TB× less expert weight traffic, the dominant prefill cost).

  static const int _grpTB = 16; // tokens per tile (also accumulators/thread)
  static const int _grpTPR = 4; // threads cooperating per row
  static const int _grpRPW = 64; // rows per workgroup (TPR*RPW = 256)

  /// Counts assignments per expert, prefix-sums group offsets, scatters the
  /// sorted assignment list and emits the compact (expert, tileStart)
  /// worklist — ONE workgroup, one thread per expert (E <= 256 as topkB
  /// already assumes), NO atomics: FXC E_FAILs on any workgroup atomic
  /// array (probed), so each expert-thread just scans all T*K assignments
  /// (broadcast reads, trivial) and scatter order is deterministic
  /// (ascending z within each group).
  void _fireExpGroupPrep(int experts, int T) {
    final prep = _sh('''
// plan-expgroupPrep
@group(0) @binding(0) var<storage, read_write> idxb: array<u32>;
@group(0) @binding(1) var<storage, read_write> pT: array<u32>;
@group(0) @binding(2) var<storage, read_write> off: array<u32>;
@group(0) @binding(3) var<storage, read_write> wl: array<u32>;
@group(0) @binding(4) var<storage, read_write> wlc: array<u32>;
@group(0) @binding(5) var<storage, read_write> srt: array<u32>;

const E: u32 = ${experts}u;
const K: u32 = ${topK}u;
const TB: u32 = ${_grpTB}u;

var<workgroup> scnt: array<u32, $experts>;
var<workgroup> soff: array<u32, $experts>;

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
    prep.setBuffer('idxb', _pTopkIdx!);
    prep.setBuffer('pT', _pT!);
    prep.setBuffer('off', _pExpOff!);
    prep.setBuffer('wl', _pWl!);
    prep.setBuffer('wlc', _pWlc!);
    prep.setBuffer('srt', _pSorted!);
    prep.dispatchFire(1, 1, 1);
  }

  /// Grouped expert matvec over one worklist tile per workgroup-y: 256
  /// threads = TPR per row × RPW rows; each quant word loaded once serves
  /// all TB tokens of the tile.  The block loop runs in ROUNDS of TPR
  /// blocks: each round cooperatively STAGES the tile's x slice
  /// (TPR blocks × 8 words × TB tokens of vec4s) into shared memory, so
  /// per-quant-word x traffic is shared reads instead of TB storage loads
  /// per thread (RPW rows re-read the same x otherwise).  All barriers run
  /// unconditionally (empty tiles stage garbage-but-valid x and skip the
  /// math), so no uniformity tricks are needed.
  /// With [fuseSiluU], [x] is the gate activations and the staging phase
  /// computes silu(g)·u inline (down-projection fusion — kills the separate
  /// SiLU dispatch and the prodExp round-trip).
  void _fireExpGrouped(QuantizedTensor stack, String tag, Buffer x, Buffer y,
      int T, int experts, bool xPerZ,
      {Buffer? fuseSiluU}) {
    final traits = ggmlTypeTraits[stack.type]!;
    final bpe = stack.rows * stack.cols ~/ traits.blockSize * traits.typeSize;
    final silu = fuseSiluU != null;
    // The per-token accumulators are UNROLLED into scalars (generated here):
    // a dynamically-indexed `array<f32, TB>` local forces FXC into indexable
    // temp registers (effectively local memory) and halves throughput.
    final jt = List.generate(_grpTB, (i) => i);
    final accDecl = jt.map((i) => '  var acc$i: f32 = 0.0;').join('\n');
    final bsumDecl =
        jt.map((i) => '      var bsum$i: f32 = 0.0;').join('\n');
    // xs layout [tp][jt][k] (vec4 units): compute reads broadcast across
    // rows; staging writes are linear (conflict-free).
    final macs = jt
        .map((i) =>
            '          bsum$i = bsum$i + dot(qv, xs[xsb + ${i * 8}u + k]);')
        .join('\n');
    final accAdd =
        jt.map((i) => '        acc$i = acc$i + d * bsum$i;').join('\n');
    final redStore = jt
        .map((i) =>
            '  red[(lid.y * TB + ${i}u) * TPR + lid.x] = acc$i;')
        .join('\n');
    final s = _qmvShaders.putIfAbsent(
        'Bgrp$tag${silu ? 's' : ''}${xPerZ ? 'z' : ''}'
        '/${stack.rows}x${stack.cols}/t${stack.type}',
        () => _sh('''
// plan-expgroupB:$tag${silu ? ':silu' : ''}
@group(0) @binding(0) var<storage, read_write> wq: array<u32>;
@group(0) @binding(1) var<storage, read_write> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> srt: array<u32>;
@group(0) @binding(4) var<storage, read_write> off: array<u32>;
@group(0) @binding(5) var<storage, read_write> wl: array<u32>;
@group(0) @binding(6) var<storage, read_write> wlc: array<u32>;
${silu ? '@group(0) @binding(7) var<storage, read_write> u4: array<vec4<f32>>;' : ''}

const ROWS: u32 = ${stack.rows}u;
const COLS: u32 = ${stack.cols}u;
const K: u32 = ${topK}u;
const TB: u32 = ${_grpTB}u;
const TPR: u32 = ${_grpTPR}u;
const RPW: u32 = ${_grpRPW}u;

${QuantizedTensor.accessorsWGSL}

var<workgroup> zs: array<u32, $_grpTB>;
var<workgroup> xb: array<u32, $_grpTB>;
var<workgroup> xs: array<vec4<f32>, ${_grpTPR * 8 * _grpTB}>;
var<workgroup> red: array<f32, ${_grpTB * _grpTPR * _grpRPW}>;

@compute @workgroup_size($_grpTPR, $_grpRPW)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  // Tile metadata: identical on every thread, so barriers stay uniform.
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
    xb[tid] = ${xPerZ ? 'z' : '(z / K)'} * COLS / 4u;
  }
  workgroupBarrier();
  let row: u32 = wid.x * RPW + lid.y;
  let eb: u32 = e * ${bpe}u;
  let nb: u32 = COLS / 32u;
  let xsb: u32 = lid.x * ${_grpTB * 8}u;
$accDecl
  for (var r: u32 = 0u; r < ${(stack.cols ~/ 32 + _grpTPR - 1) ~/ _grpTPR}u; r = r + 1u) {
    // Stage this round's x slice: [tp][jt][k] vec4s, linear writes.
    for (var si: u32 = tid; si < ${_grpTPR * 8 * _grpTB}u; si = si + ${_grpTPR * _grpRPW}u) {
      let jj: u32 = r * TPR + si / ${_grpTB * 8}u;
      if (jj < nb) {
        let a: u32 = xb[(si / 8u) % TB] + jj * 8u + (si % 8u);
${silu ? '''
        let gv: vec4<f32> = x[a];
        xs[si] = (gv / (vec4<f32>(1.0) + exp(-gv))) * u4[a];
''' : '''
        xs[si] = x[a];
'''}
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
  // Threads sweep the RPW*TB (rowInWg, token) pairs; TPR partials each.
  for (var pp: u32 = tid; pp < RPW * TB; pp = pp + ${_grpTPR * _grpRPW}u) {
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
'''));
    s.setBuffer('wq', stack.buffer);
    s.setBuffer('x', x);
    s.setBuffer('y', y);
    s.setBuffer('srt', _pSorted!);
    s.setBuffer('off', _pExpOff!);
    s.setBuffer('wl', _pWl!);
    s.setBuffer('wlc', _pWlc!);
    if (silu) s.setBuffer('u4', fuseSiluU);
    final rowGroups = (stack.rows + _grpRPW - 1) ~/ _grpRPW;
    // Strict worklist bound: sum(ceil(cnt_e/TB)) <= ceil(T*K/TB) + E.
    final wlBound = (T * topK + _grpTB - 1) ~/ _grpTB + experts;
    s.dispatchFire(rowGroups, wlBound, 1);
  }

  /// dot4I8Packed variant of [_fireExpGrouped]: activations arrive
  /// PRE-QUANTIZED to int8 ([xq] packed words + [xsc] per-32-block scales),
  /// staging holds packed u32 quants (4x smaller than the vec4 f32 stage),
  /// and each weight word meets each token's word in ONE dot4I8Packed —
  /// the grouped kernels do TB=16 MACs per weight byte, the arithmetic-
  /// heavy regime where the int8 MAC is microbench-proven.  Structure
  /// (metadata / rounds / unrolled per-token accumulators / TPR-wide
  /// reduction) is identical to the f32 variant.
  void _fireExpGroupedQ(QuantizedTensor stack, String tag, Buffer xq,
      Buffer xsc, Buffer y, int T, int experts, bool xPerZ) {
    final traits = ggmlTypeTraits[stack.type]!;
    final bpe = stack.rows * stack.cols ~/ traits.blockSize * traits.typeSize;
    final jt = List.generate(_grpTB, (i) => i);
    final accDecl = jt.map((i) => '  var acc$i: f32 = 0.0;').join('\n');
    final isumDecl =
        jt.map((i) => '      var isum$i: i32 = 0;').join('\n');
    final macs = jt
        .map((i) =>
            '        isum$i = isum$i + dot4I8Packed(raw, xsq[xqb + ${i * 8}u + k]);')
        .join('\n');
    // ssb reads use literal indices into workgroup memory (cheap; hoisting
    // 16 locals would add register pressure on top of acc+isum).
    final accAdd = jt
        .map((i) =>
            '        acc$i = acc$i + d * xsc[ssb[${i}u] + j] * f32(isum$i);')
        .join('\n');
    final redStore = jt
        .map((i) =>
            '  red[(lid.y * TB + ${i}u) * TPR + lid.x] = acc$i;')
        .join('\n');
    final s = _qmvShaders.putIfAbsent(
        'BgrpQ$tag${xPerZ ? 'z' : ''}/${stack.rows}x${stack.cols}/t${stack.type}',
        () => _sh('''
// plan-expgroupQ:$tag
@group(0) @binding(0) var<storage, read_write> wq: array<u32>;
@group(0) @binding(1) var<storage, read_write> xq: array<u32>;
@group(0) @binding(2) var<storage, read_write> xsc: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read_write> srt: array<u32>;
@group(0) @binding(5) var<storage, read_write> off: array<u32>;
@group(0) @binding(6) var<storage, read_write> wl: array<u32>;
@group(0) @binding(7) var<storage, read_write> wlc: array<u32>;

const ROWS: u32 = ${stack.rows}u;
const COLS: u32 = ${stack.cols}u;
const K: u32 = ${topK}u;
const TB: u32 = ${_grpTB}u;
const TPR: u32 = ${_grpTPR}u;
const RPW: u32 = ${_grpRPW}u;

${QuantizedTensor.accessorsWGSL}

var<workgroup> zs: array<u32, $_grpTB>;
var<workgroup> swb: array<u32, $_grpTB>;
var<workgroup> ssb: array<u32, $_grpTB>;
var<workgroup> xsq: array<u32, ${_grpTPR * 8 * _grpTB}>;
var<workgroup> red: array<f32, ${_grpTB * _grpTPR * _grpRPW}>;

@compute @workgroup_size($_grpTPR, $_grpRPW)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  // Tile metadata: identical on every thread, so barriers stay uniform.
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
    let xrow: u32 = ${xPerZ ? 'z' : '(z / K)'};
    swb[tid] = xrow * (COLS / 4u);
    ssb[tid] = xrow * (COLS / 32u);
  }
  workgroupBarrier();
  let row: u32 = wid.x * RPW + lid.y;
  let eb: u32 = e * ${bpe}u;
  let nb: u32 = COLS / 32u;
  let xqb: u32 = lid.x * ${_grpTB * 8}u;
$accDecl
  for (var r: u32 = 0u; r < ${(stack.cols ~/ 32 + _grpTPR - 1) ~/ _grpTPR}u; r = r + 1u) {
    // Stage this round's packed int8 activations: [tp][jt][k] u32 words.
    for (var si: u32 = tid; si < ${_grpTPR * 8 * _grpTB}u; si = si + ${_grpTPR * _grpRPW}u) {
      let jj: u32 = r * TPR + si / ${_grpTB * 8}u;
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
  // Threads sweep the RPW*TB (rowInWg, token) pairs; TPR partials each.
  for (var pp: u32 = tid; pp < RPW * TB; pp = pp + ${_grpTPR * _grpRPW}u) {
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
'''));
    s.setBuffer('wq', stack.buffer);
    s.setBuffer('xq', xq);
    s.setBuffer('xsc', xsc);
    s.setBuffer('y', y);
    s.setBuffer('srt', _pSorted!);
    s.setBuffer('off', _pExpOff!);
    s.setBuffer('wl', _pWl!);
    s.setBuffer('wlc', _pWlc!);
    final rowGroups = (stack.rows + _grpRPW - 1) ~/ _grpRPW;
    final wlBound = (T * topK + _grpTB - 1) ~/ _grpTB + experts;
    s.dispatchFire(rowGroups, wlBound, 1);
  }

  /// DENSE int8 GEMM: y[t*ROWS+row] = W @ x[t] over a chunk of T token
  /// rows, weights q8_0, activations pre-quantized int8 ([xq]/[xsc]).
  /// Replaces the dequant-to-f16 + f16-GEMM path for the dense prefill
  /// projections: no f16 scratch pass at all, and the MACs are hardware
  /// int8.  Same skeleton as [_fireExpGroupedQ] minus the expert
  /// indirection — token tiles are CONTIGUOUS (tile base wid.y*TB), the
  /// tail tile bound comes from pT[0].
  void _fireGemmQ(QuantizedTensor w, String tag, Buffer xq, Buffer xsc,
      Buffer y, int T) {
    final jt = List.generate(_grpTB, (i) => i);
    final accDecl = jt.map((i) => '  var acc$i: f32 = 0.0;').join('\n');
    final isumDecl =
        jt.map((i) => '      var isum$i: i32 = 0;').join('\n');
    final macs = jt
        .map((i) =>
            '        isum$i = isum$i + dot4I8Packed(raw, xsq[xqb + ${i * 8}u + k]);')
        .join('\n');
    final accAdd = jt
        .map((i) =>
            '        acc$i = acc$i + d * xsc[(t0 + ${i}u) * (COLS / 32u) + j] * f32(isum$i);')
        .join('\n');
    final redStore = jt
        .map((i) =>
            '  red[(lid.y * TB + ${i}u) * TPR + lid.x] = acc$i;')
        .join('\n');
    final s = _qmvShaders.putIfAbsent(
        'GemmQ$tag/${w.rows}x${w.cols}', () => _sh('''
// plan-gemmq:$tag
@group(0) @binding(0) var<storage, read_write> wq: array<u32>;
@group(0) @binding(1) var<storage, read_write> xq: array<u32>;
@group(0) @binding(2) var<storage, read_write> xsc: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read_write> pT: array<u32>;

const ROWS: u32 = ${w.rows}u;
const COLS: u32 = ${w.cols}u;
const TB: u32 = ${_grpTB}u;
const TPR: u32 = ${_grpTPR}u;
const RPW: u32 = ${_grpRPW}u;

${QuantizedTensor.accessorsWGSL}

var<workgroup> xsq: array<u32, ${_grpTPR * 8 * _grpTB}>;
var<workgroup> red: array<f32, ${_grpTB * _grpTPR * _grpRPW}>;

@compute @workgroup_size($_grpTPR, $_grpRPW)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let t0: u32 = wid.y * TB;
  let n: u32 = min(TB, pT[0] - min(t0, pT[0]));
  let tid: u32 = lid.y * TPR + lid.x;
  let row: u32 = wid.x * RPW + lid.y;
  let eb: u32 = 0u;
  let nb: u32 = COLS / 32u;
  let xqb: u32 = lid.x * ${_grpTB * 8}u;
$accDecl
  for (var r: u32 = 0u; r < ${'PLACEHOLDER_ROUNDS'}u; r = r + 1u) {
    // Stage this round's packed int8 activations: [tp][jt][k] u32 words.
    // Out-of-chunk token rows stage garbage-but-in-buffer words (the xq
    // buffer is capacity-sized); their accumulators are never written.
    for (var si: u32 = tid; si < ${_grpTPR * 8 * _grpTB}u; si = si + ${_grpTPR * _grpRPW}u) {
      let jj: u32 = r * TPR + si / ${_grpTB * 8}u;
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
  for (var pp: u32 = tid; pp < RPW * TB; pp = pp + ${_grpTPR * _grpRPW}u) {
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
'''
            .replaceAll('PLACEHOLDER_ROUNDS',
                '${(w.cols ~/ 32 + _grpTPR - 1) ~/ _grpTPR}')));
    s.setBuffer('wq', w.buffer);
    s.setBuffer('xq', xq);
    s.setBuffer('xsc', xsc);
    s.setBuffer('y', y);
    s.setBuffer('pT', _pT!);
    final rowGroups = (w.rows + _grpRPW - 1) ~/ _grpRPW;
    final tokTiles = (T + _grpTB - 1) ~/ _grpTB;
    s.dispatchFire(rowGroups, tokTiles, 1);
  }

  Future<void> _moeB(PlanBlock b, int T,
      [Future<void> Function(String)? mark]) async {
    final m = b.moe;
    final experts = m.router.shape[0];

    // Router for all tokens.
    final rt = _sh('''
// plan-routerB
@group(0) @binding(0) var<storage, read_write> wf: array<f32>;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

const ROWS: u32 = ${experts}u;
const COLS: u32 = ${dim}u;

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let t: u32 = wid.z;
  let row: u32 = wid.x + wid.y * nwg.x;
  var acc: f32 = 0.0;
  if (row < ROWS) {
    for (var c: u32 = lid.x; c < COLS; c = c + 256u) {
      acc = acc + wf[row * COLS + c] * x[t * COLS + c];
    }
  }
  scratch[lid.x] = acc;
${_reduceSum(256)}
  if (lid.x == 0u && row < ROWS) { y[t * ROWS + row] = scratch[0]; }
}
''');
    rt.setBuffer('wf', m.router.buffer);
    rt.setBuffer('x', _pxn!);
    rt.setBuffer('y', _pRoutLogits!);
    _fireRowsZ(rt, experts, T);

    // Per-token top-K: one workgroup per token.
    final tk = _sh('''
// plan-topkB
@group(0) @binding(0) var<storage, read_write> lg: array<f32>;
@group(0) @binding(1) var<storage, read_write> idxout: array<u32>;
@group(0) @binding(2) var<storage, read_write> wout: array<f32>;

const E: u32 = ${experts}u;
const K: u32 = ${topK}u;

var<workgroup> p: array<f32, $experts>;
var<workgroup> sval: array<f32, 256>;
var<workgroup> sidx: array<u32, 256>;
var<workgroup> sel: array<u32, $topK>;
var<workgroup> sw: array<f32, $topK>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let t: u32 = wid.x;
  let e: u32 = lid.x;
  var v: f32 = -1.0e30;
  if (e < E) { v = lg[t * E + e]; }
  sval[lid.x] = v;
  workgroupBarrier();
  for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
    if (lid.x < s) { sval[lid.x] = max(sval[lid.x], sval[lid.x + s]); }
    workgroupBarrier();
  }
  let mx: f32 = sval[0];
  workgroupBarrier();
  if (e < E) { p[e] = exp(lg[t * E + e] - mx); }
  workgroupBarrier();

  for (var s: u32 = 0u; s < K; s = s + 1u) {
    var bv: f32 = -1.0;
    if (e < E) { bv = p[e]; }
    sval[lid.x] = bv;
    sidx[lid.x] = e;
    workgroupBarrier();
    for (var st: u32 = 128u; st > 0u; st = st >> 1u) {
      if (lid.x < st && sval[lid.x + st] > sval[lid.x]) {
        sval[lid.x] = sval[lid.x + st];
        sidx[lid.x] = sidx[lid.x + st];
      }
      workgroupBarrier();
    }
    if (lid.x == 0u) {
      sel[s] = sidx[0];
      sw[s] = sval[0];
    }
    workgroupBarrier();
    if (e == sel[s]) { p[e] = -2.0; }
    workgroupBarrier();
  }

  if (lid.x == 0u) {
    var total: f32 = 0.0;
    for (var s: u32 = 0u; s < K; s = s + 1u) { total = total + sw[s]; }
    for (var s: u32 = 0u; s < K; s = s + 1u) {
      idxout[t * K + s] = sel[s];
      wout[t * K + s] = sw[s] / total;
    }
  }
}
''');
    tk.setBuffer('lg', _pRoutLogits!);
    tk.setBuffer('idxout', _pTopkIdx!);
    tk.setBuffer('wout', _pTopkW!);
    tk.dispatchFire(T, 1, 1);
    if (mark != null) await mark('moe:route');

    final gk = m.stackGate!, uk = m.stackUp!, dk = m.stackDown!;
    final grouped = experts <= 256 &&
        gk.type == GgmlType.q8_0 &&
        uk.type == GgmlType.q8_0 &&
        dk.type == GgmlType.q8_0 &&
        !const bool.fromEnvironment('GPU_ML_NO_GROUP');
    // int8-quantize the post-norm rows ONCE; shexp gate/up (dense int8
    // GEMM) and the grouped expert kernels all reuse it.
    final gq = prefillDp4a && _pXnq != null;
    if (gq) {
      _fireQuantXB('pxn', _pxn!, maxPrefillChunk * dim, _pXnq!, _pXnsc!);
    }

    // The grouped path's combine WRITES the full ffn sum (experts + shared
    // expert), so it needs no zero pass and the shared-expert down goes
    // through the fast GEMM path into its own scratch.
    if (!grouped) {
      _fireZeroN('pffn', _pffn!, T * dim);
    }

    // Shared expert: sigmoid(dot) per token, then batched FFN accumulate.
    if (m.gateShexp != null) {
      final sg = _sh('''
// plan-shscalarB
@group(0) @binding(0) var<storage, read_write> w: array<f32>;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> outs: array<f32>;

const N: u32 = ${dim}u;

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let t: u32 = wid.x;
  var acc: f32 = 0.0;
  for (var i: u32 = lid.x; i < N; i = i + 256u) {
    acc = acc + w[i] * x[t * N + i];
  }
  scratch[lid.x] = acc;
${_reduceSum(256)}
  if (lid.x == 0u) { outs[t] = 1.0 / (1.0 + exp(-scratch[0])); }
}
''');
      sg.setBuffer('w', m.sharedGate!.buffer);
      sg.setBuffer('x', _pxn!);
      sg.setBuffer('outs', _pShScalar!);
      sg.dispatchFire(T, 1, 1);

      _fireQmvB(m.gateShexp!, 'sh_gate', _pxn!, _pgSh!, T,
          xq: gq ? _pXnq : null, xsc: gq ? _pXnsc : null);
      _fireQmvB(m.upShexp!, 'sh_up', _pxn!, _puSh!, T,
          xq: gq ? _pXnq : null, xsc: gq ? _pXnsc : null);
      _fireSiluMulN('psh', _pgSh!, _puSh!, _pprodSh!, T * m.gateShexp!.rows);
      final dsh = m.downShexp!;
      if (grouped) {
        // GEMM down into its own scratch; the sigmoid scale + ffn sum land
        // in the combine kernel.  (The accumulate variant below launches a
        // workgroup per (row, token) — 524k workgroups per block.)
        _fireQmvB(dsh, 'sh_down', _pprodSh!, _pshDown!, T);
      } else {
      // down accumulate with per-token sigmoid scale.
      final s = _qmvShaders.putIfAbsent(
          'Bsh/${dsh.rows}x${dsh.cols}/t${dsh.type}', () => _sh('''
// plan-shdownB
@group(0) @binding(0) var<storage, read_write> wq: array<u32>;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> wsel: array<f32>;

const ROWS: u32 = ${dsh.rows}u;
const COLS: u32 = ${dsh.cols}u;

${QuantizedTensor.accessorsWGSL}
${QuantizedTensor.typeNeedsScaleMinK4(dsh.type) ? QuantizedTensor.scaleMinK4WGSL : ''}

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let t: u32 = wid.z;
  let row: u32 = wid.x + wid.y * nwg.x;
  let eb: u32 = 0u;
  let xbase: u32 = t * COLS;
  var acc: f32 = 0.0;
  if (row < ROWS) {
${_matVecBodyOffsetX(dsh.type)}
  }
  scratch[lid.x] = acc;
${_reduceSum(256)}
  if (lid.x == 0u && row < ROWS) {
    y[t * ROWS + row] = y[t * ROWS + row] + wsel[t] * scratch[0];
  }
}
'''));
      s.setBuffer('wq', dsh.buffer);
      s.setBuffer('x', _pprodSh!);
      s.setBuffer('y', _pffn!);
      s.setBuffer('wsel', _pShScalar!);
      _fireRowsZ(s, dsh.rows, T);
      }
    }
    if (mark != null) await mark('moe:shexp');

    // Fused experts: z = t*K + slot.
    void fused(QuantizedTensor stack, String tag, Buffer x, Buffer y,
        bool xPerZ) {
      final traits = ggmlTypeTraits[stack.type]!;
      final bpe = stack.rows * stack.cols ~/ traits.blockSize * traits.typeSize;
      final s = _qmvShaders.putIfAbsent(
          'Bexp$tag/${stack.rows}x${stack.cols}/t${stack.type}', () => _sh('''
// plan-expfusedB:$tag
@group(0) @binding(0) var<storage, read_write> wq: array<u32>;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> idxb: array<u32>;

const ROWS: u32 = ${stack.rows}u;
const COLS: u32 = ${stack.cols}u;
const K: u32 = ${topK}u;

${QuantizedTensor.accessorsWGSL}
${QuantizedTensor.typeNeedsScaleMinK4(stack.type) ? QuantizedTensor.scaleMinK4WGSL : ''}

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let z: u32 = wid.z;               // t * K + slot
  let row: u32 = wid.x + wid.y * nwg.x;
  let eb: u32 = idxb[z] * ${bpe}u;
  let xbase: u32 = ${xPerZ ? 'z' : '(z / K)'} * COLS;
  var acc: f32 = 0.0;
  if (row < ROWS) {
${_matVecBodyOffsetX(stack.type)}
  }
  scratch[lid.x] = acc;
${_reduceSum(256)}
  if (lid.x == 0u && row < ROWS) { y[z * ROWS + row] = scratch[0]; }
}
'''));
      s.setBuffer('wq', stack.buffer);
      s.setBuffer('x', x);
      s.setBuffer('y', y);
      s.setBuffer('idxb', _pTopkIdx!);
      _fireRowsZ(s, stack.rows, T * topK);
    }

    if (grouped && gq) {
      // dp4a experts: gate/up/down as int8-MAC grouped kernels, reusing
      // the block's hoisted _pXnq quantization.  The silu stays a separate
      // pass here (its output must be re-quantized globally per 32-block
      // before down — scales can't be computed inside the down kernel's
      // scattered staging).
      _fireExpGroupPrep(experts, T);
      if (mark != null) await mark('moe:prep');
      _fireExpGroupedQ(gk, 'gate', _pXnq!, _pXnsc!, _pgExp!, T, experts, false);
      if (mark != null) await mark('moe:gate');
      _fireExpGroupedQ(uk, 'up', _pXnq!, _pXnsc!, _puExp!, T, experts, false);
      if (mark != null) await mark('moe:up');
      _fireSiluMulN('pexp', _pgExp!, _puExp!, _pprodExp!, T * topK * gk.rows);
      _fireQuantXB('prod', _pprodExp!, maxPrefillChunk * topK * gk.rows,
          _pProdq!, _pProdsc!);
      _fireExpGroupedQ(
          dk, 'down', _pProdq!, _pProdsc!, _pdownExp!, T, experts, true);
      if (mark != null) await mark('moe:down');
    } else if (grouped) {
      _fireExpGroupPrep(experts, T);
      if (mark != null) await mark('moe:prep');
      _fireExpGrouped(gk, 'gate', _pxn!, _pgExp!, T, experts, false);
      if (mark != null) await mark('moe:gate');
      _fireExpGrouped(uk, 'up', _pxn!, _puExp!, T, experts, false);
      if (mark != null) await mark('moe:up');
      // SiLU·mul is FUSED into the down kernel's staging phase (one less
      // dispatch + no prodExp round-trip).
      _fireExpGrouped(dk, 'down', _pgExp!, _pdownExp!, T, experts, true,
          fuseSiluU: _puExp!);
      if (mark != null) await mark('moe:down');
    } else {
      fused(gk, 'gate', _pxn!, _pgExp!, false);
      fused(uk, 'up', _pxn!, _puExp!, false);
      _fireSiluMulN('pexp', _pgExp!, _puExp!, _pprodExp!, T * topK * gk.rows);
      fused(dk, 'down', _pprodExp!, _pdownExp!, true);
    }

    // Combine.  Grouped path: ffn[t] = shScalar[t]*shDown[t] +
    // sum_slot w[t,slot]*down[t,slot] — a WRITE folding the shared-expert
    // accumulate, so no zero pass is needed.  Fallback: RMW into the
    // zeroed+accumulated ffn.
    final withSh = grouped && m.gateShexp != null;
    final cb = _sh('''
// plan-expcombineB${withSh ? ':sh' : grouped ? ':w' : ''}
@group(0) @binding(0) var<storage, read_write> ydown: array<f32>;
@group(0) @binding(1) var<storage, read_write> wsel: array<f32>;
@group(0) @binding(2) var<storage, read_write> ffn: array<f32>;
${withSh ? '''
@group(0) @binding(3) var<storage, read_write> shd: array<f32>;
@group(0) @binding(4) var<storage, read_write> shsc: array<f32>;
''' : ''}

const DIM: u32 = ${dim}u;
const K: u32 = ${topK}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i >= ${maxPrefillChunk * dim}u) { return; }
  let t: u32 = i / DIM;
  let row: u32 = i % DIM;
  var acc: f32 = 0.0;
  for (var s: u32 = 0u; s < K; s = s + 1u) {
    acc = acc + wsel[t * K + s] * ydown[(t * K + s) * DIM + row];
  }
${withSh ? '  ffn[i] = shsc[t] * shd[i] + acc;' : grouped ? '  ffn[i] = acc;' : '  ffn[i] = ffn[i] + acc;'}
}
''');
    cb.setBuffer('ydown', _pdownExp!);
    cb.setBuffer('wsel', _pTopkW!);
    cb.setBuffer('ffn', _pffn!);
    if (withSh) {
      cb.setBuffer('shd', _pshDown!);
      cb.setBuffer('shsc', _pShScalar!);
    }
    _fireLinear(cb, T * dim);
    if (mark != null) await mark('moe:comb');
  }

  void _fireZeroN(String tag, Buffer y, int n) {
    final s = _sh('''
// plan-zeroN:$tag
@group(0) @binding(0) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  y[i] = 0.0;
}
''');
    s.setBuffer('y', y);
    _fireLinear(s, n);
  }

  void _fireSiluMulN(String tag, Buffer g, Buffer u, Buffer prod, int n) {
    final s = _sh('''
// plan-silumulN:$tag
@group(0) @binding(0) var<storage, read_write> g: array<f32>;
@group(0) @binding(1) var<storage, read_write> u: array<f32>;
@group(0) @binding(2) var<storage, read_write> prod: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  let gv: f32 = g[i];
  prod[i] = (gv / (1.0 + exp(-gv))) * u[i];
}
''');
    s.setBuffer('g', g);
    s.setBuffer('u', u);
    s.setBuffer('prod', prod);
    _fireLinear(s, n);
  }

  /// Runs a prefill chunk of [T] tokens whose embeddings are in [rows]
  /// (T*dim floats), starting at position [startPos].  Fully fired; leaves
  /// the LAST token's hidden state in the decode x buffer so [readLogits] /
  /// [readX] work as after a decode step.
  Future<void> prefillChunk(Float32List rows, int T, int startPos) async {
    await _prefillStart(T, startPos);
    await _px!.write(rows, T * dim);
    await _prefillBody(T);
  }

  /// Batched prefill fed by TOKEN IDS: embedding rows are gathered and
  /// dequantized on-GPU from the resident [embed] table, replacing one
  /// awaited ~2KB disk read per prompt token (which was 1-2s of wall time
  /// per 512-token prompt).
  Future<void> prefillChunkTokens(
      QuantizedTensor embed, List<int> tokens, int startPos) async {
    final T = tokens.length;
    await _prefillStart(T, startPos);
    final tok = Uint32List(T);
    for (int i = 0; i < T; i++) {
      tok[i] = tokens[i];
    }
    await _pTok!.write(tok, T, dataType: BufferDataType.uint32);
    _fireEmbedGatherB(embed, T);
    await _prefillBody(T);
  }

  Future<void> _prefillStart(int T, int startPos) async {
    if (!supportsBatchedPrefill) {
      throw Exception('batched prefill requires all blocks resident');
    }
    if (T < 4 || T > maxPrefillChunk) {
      throw Exception('prefill chunk T=$T out of range 4..$maxPrefillChunk');
    }
    if (startPos + T > maxSeq) {
      throw Exception('prefill exceeds KV capacity $maxSeq');
    }
    _ensurePrefillBuffers();

    // Drain this device's task FIFO before touching shared inputs: queue
    // writes execute at SUBMISSION time, so without this they jump ahead of
    // any still-queued fired dispatches from the previous chunk — which
    // read _pos/_pT and would see the NEW chunk's values.  (Decode never
    // needs this because every token ends in an awaited readback.)
    {
      final probe = Float32List(1);
      final sw = Stopwatch()..start();
      await _x.read(probe, 1);
      syncMicros += sw.elapsedMicroseconds;
    }

    final posData = Uint32List(4)..[0] = startPos;
    await _pos.write(posData, 4, dataType: BufferDataType.uint32);
    _lastGpuPos = -2; // decode must re-seed _pos after a prefill chunk
    final tData = Uint32List(4)..[0] = T;
    await _pT!.write(tData, 4, dataType: BufferDataType.uint32);
  }

  /// GPU embed gather: one workgroup-z per token, dequantizing that token's
  /// row straight into the prefill x buffer.
  void _fireEmbedGatherB(QuantizedTensor embed, int T,
      {Buffer? tok, Buffer? out}) {
    tok ??= _pTok!;
    out ??= _px!;
    final ComputeShader s;
    if (embed.type == GgmlType.q8_0) {
      final nb = dim ~/ 32;
      s = _sh('''
// plan-embgather:q8
@group(0) @binding(0) var<storage, read_write> wq: array<u32>;
@group(0) @binding(1) var<storage, read_write> tok: array<u32>;
@group(0) @binding(2) var<storage, read_write> px: array<f32>;

const D: u32 = ${dim}u;
const NB: u32 = ${nb}u;

${QuantizedTensor.accessorsWGSL}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let t: u32 = wid.z;
  let j: u32 = lid.x + wid.x * 256u;
  if (j >= NB) { return; }
  let base: u32 = tok[t] * ${nb * 34}u + j * 34u;
  let d: f32 = f16At(base);
  let qb: u32 = base + 2u;
  let qw: u32 = qb >> 2u;
  let qs: u32 = (qb & 3u) * 8u;
  var carry: u32 = wq[qw];
  let ob: u32 = t * D + j * 32u;
  for (var k: u32 = 0u; k < 8u; k = k + 1u) {
    let nxt: u32 = wq[qw + k + 1u];
    var raw: u32;
    if (qs == 0u) { raw = carry; } else { raw = (carry >> qs) | (nxt << (32u - qs)); }
    carry = nxt;
    px[ob + k * 4u] = d * f32(bitcast<i32>(raw << 24u) >> 24u);
    px[ob + k * 4u + 1u] = d * f32(bitcast<i32>(raw << 16u) >> 24u);
    px[ob + k * 4u + 2u] = d * f32(bitcast<i32>(raw << 8u) >> 24u);
    px[ob + k * 4u + 3u] = d * f32(bitcast<i32>(raw) >> 24u);
  }
}
''');
    } else if (embed.type == GgmlType.f16) {
      s = _sh('''
// plan-embgather:f16
@group(0) @binding(0) var<storage, read_write> wq: array<u32>;
@group(0) @binding(1) var<storage, read_write> tok: array<u32>;
@group(0) @binding(2) var<storage, read_write> px: array<f32>;

const D: u32 = ${dim}u;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let t: u32 = wid.z;
  let w: u32 = lid.x + wid.x * 256u;
  if (w >= D / 2u) { return; }
  let pair = unpack2x16float(wq[tok[t] * (D / 2u) + w]);
  px[t * D + w * 2u] = pair.x;
  px[t * D + w * 2u + 1u] = pair.y;
}
''');
    } else {
      throw Exception('embed gather unsupported for type ${embed.type}');
    }
    s.setBuffer('wq', embed.buffer);
    s.setBuffer('tok', tok);
    s.setBuffer('px', out);
    final work = embed.type == GgmlType.q8_0 ? dim ~/ 32 : dim ~/ 2;
    s.dispatchFire((work + 255) ~/ 256, 1, T);
  }

  Buffer? _tokDec;

  /// DECODE-side GPU embed gather: writes the token id (4 bytes) and
  /// dequantizes its embedding row straight into the decode x buffer —
  /// replaces the per-token ~2KB DISK read + CPU dequant + 8KB upload of
  /// the _embedRow path.  Safe to fire immediately: decode's previous
  /// readback drained all pending work, same contract as writeX.
  Future<void> loadTokenX(QuantizedTensor embed, int token) async {
    _tokDec ??= _u32(4);
    final t = Uint32List(4)..[0] = token;
    await _tokDec!.write(t, 4, dataType: BufferDataType.uint32);
    _fireEmbedGatherB(embed, 1, tok: _tokDec!, out: _x);
  }

  /// One single-device decode step fed by TOKEN ID (GPU embed gather).
  Future<Float32List> forwardToken(
      QuantizedTensor embed, int token, int pos) async {
    await loadTokenX(embed, token);
    await runBlocks(pos);
    return readLogits();
  }

  /// Greedy decode step by token id: GPU embed gather + blocks + GPU argmax;
  /// returns the argmax'd NEXT token id via a 4-byte readback.
  Future<int> forwardTokenGreedy(
      QuantizedTensor embed, int token, int pos) async {
    await loadTokenX(embed, token);
    await runBlocks(pos);
    return readArgmaxToken();
  }

  /// Greedy decode step CHAINED from the previous [readArgmaxToken]: the
  /// embed gather reads the token id straight from the GPU argmax buffer,
  /// so the whole token is fired dispatches + ONE 4-byte readback — zero
  /// CPU-side queue writes.  Only valid immediately after a greedy step
  /// (the caller must pass the id that step returned to its sampler/EOS
  /// logic; [_amaxTok] still holds it).
  Future<int> forwardChainGreedy(QuantizedTensor embed, int pos) async {
    _fireEmbedGatherB(embed, 1, tok: _amaxTok!, out: _x);
    await runBlocks(pos);
    return readArgmaxToken();
  }

  Buffer? _tokHist;

  /// K-token GPU-RESIDENT greedy decode: fires [k] chained token steps
  /// (embed gather from the argmax buffer -> blocks -> head -> argmax ->
  /// id into slot i of a history buffer) with ONE readback of the k ids at
  /// the end.  Must follow a greedy step (chain seeded).  NOTE: if EOS
  /// appears mid-burst the recurrent state and KV advance past it with
  /// post-EOS tokens — callers must stop at EOS and not resume from this
  /// state expecting pre-EOS contents (fine for generation-ending).
  Future<Uint32List> forwardBurstGreedy(
      QuantizedTensor embed, int posStart, int k) async {
    if (k < 1 || k > 64) throw Exception('burst k=$k out of range 1..64');
    _tokHist ??= _u32(64);
    for (int i = 0; i < k; i++) {
      _fireEmbedGatherB(embed, 1, tok: _amaxTok!, out: _x);
      await runBlocks(posStart + i);
      _fireHead();
      _fireArgmax();
      final cp = _sh('''
// plan-tokcopy:$i
@group(0) @binding(0) var<storage, read_write> t: array<u32>;
@group(0) @binding(1) var<storage, read_write> h: array<u32>;
@compute @workgroup_size(1)
fn main() { h[${i}u] = t[0]; }
''');
      cp.setBuffer('t', _amaxTok!);
      cp.setBuffer('h', _tokHist!);
      cp.dispatchFire(1, 1, 1);
    }
    final out = Uint32List(64);
    final sw = Stopwatch()..start();
    await _tokHist!.read(out, k, dataType: BufferDataType.uint32);
    syncMicros += sw.elapsedMicroseconds;
    return out;
  }

  /// Cumulative CPU microseconds spent recording prefill dispatches/binds
  /// (the fire loop has no awaits, so this is pure CPU time).
  int recordMicros = 0;

  Future<void> _prefillBody(int T) async {
    final recSw = Stopwatch()..start();
    // Stage profiling (compile-time flag): drain the FIFO between stages
    // and bucket wall time.  The drains serialize the pipeline, so absolute
    // totals inflate — use the proportions.
    const prof = bool.fromEnvironment('GPU_ML_PREFILL_PROFILE');
    Future<void> mark(String k) async {
      if (!prof) return;
      final probe = Float32List(1);
      final sw = Stopwatch()..start();
      await _x.read(probe, 1);
      _profBuckets[k] = (_profBuckets[k] ?? 0) + sw.elapsedMicroseconds;
    }

    for (int i = 0; i < blocks.length; i++) {
      final b = blocks[i];
      final st = _blockState[i];
      _fireRmsB('attn_norm', _px!, b.attnNorm.buffer, _pxn!, T);
      if (prof) await mark('norm');
      if (b.delta != null) {
        _fireDeltaB(b, st, T);
        if (prof) await mark('delta');
      } else {
        _fireAttnB(b, st, T);
        if (prof) await mark('attn');
      }
      _fireAddB('res_attn', _px!, _paOut!, T * dim);
      _fireRmsB('post_norm', _px!, b.postNorm.buffer, _pxn!, T);
      if (prof) await mark('norm');
      await _moeB(b, T, prof ? mark : null);
      if (prof) await mark('moe:tail');
      _fireAddB('res_ffn', _px!, _pffn!, T * dim);
    }
    if (prof) {
      await mark('tail');
      final parts = _profBuckets.entries
          .map((e) =>
              '${e.key}=${(e.value / 1000).toStringAsFixed(1)}ms')
          .join(' ');
      // ignore: avoid_print
      print('prefill-profile T=$T: $parts');
      _profBuckets.clear();
    }

    // Copy the last token's hidden state into the decode x buffer.
    final cp = _sh('''
// plan-plastx
@group(0) @binding(0) var<storage, read_write> px: array<f32>;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> pT: array<u32>;

const D: u32 = ${dim}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i >= D) { return; }
  x[i] = px[(pT[0] - 1u) * D + i];
}
''');
    cp.setBuffer('px', _px!);
    cp.setBuffer('x', _x);
    cp.setBuffer('pT', _pT!);
    _fireLinear(cp, dim);
    recordMicros += recSw.elapsedMicroseconds;
  }

  /// Zeroes every DeltaNet block's recurrent state (ssm state + conv
  /// history) and drains.  Used after the load-time pipeline warmup chunk
  /// so its garbage tokens leave no trace; attention KV needs no reset
  /// (positions are overwritten before they are ever attended).
  Future<void> zeroRecurrentState() async {
    for (final b in blocks) {
      final d = b.delta;
      if (d == null) continue;
      _fireZeroN('convstate', d.convState.buffer, d.convState.size);
      _fireZeroN('ssmstate', d.ssmState.buffer, d.ssmState.size);
    }
    final probe = Float32List(1);
    await _x.read(probe, 1);
  }

  /// Reads the whole chunk's hidden states (multi-GPU prefill hop).
  Future<Float32List> readPrefillX(int T) async {
    final out = Float32List(T * dim);
    final sw = Stopwatch()..start();
    await _px!.read(out, T * dim);
    syncMicros += sw.elapsedMicroseconds;
    return out;
  }

  void destroy() {
    for (final s in _shaders.values) {
      s.destroy();
    }
    _shaders.clear();
    _qmvShaders.clear();
    for (final st in _blockState) {
      st.kCache?.destroy();
      st.vCache?.destroy();
      st.deltaConsts?.destroy();
    }
    _blockState.clear();
    for (final buf in [
      _x, _xn, _aOut, _ffnOut, _pos, _logits, _routLogits, _topkIdx, _topkW,
      _qFull, _kRaw, _vRaw, _q, _gate, _gated, _scores,
      _qkv, _z, _betaRaw, _alphaRaw, _dParams, _convOut, _qn, _kn, _core,
      _dGated,
      _gExp, _uExp, _prodExp, _gSh, _uSh, _prodSh, _shScalar,
      _gExpAll, _uExpAll, _prodExpAll, _downExpAll,
      _guExpAll, _guSh, _qkvz, _shOut,
      _amaxV, _amaxI, _amaxTok, _tokHist,
      _xnq, _xnsc, _prodq, _prodsc,
      _pT, _px, _pxn, _paOut, _pffn,
      _pqFull, _pkRaw, _pvRaw, _pq, _pgate, _pgated,
      _pqkv, _pz, _pbetaRaw, _palphaRaw, _pdParams, _pconvOut, _pqn, _pkn,
      _pcore, _pdGated,
      _pRoutLogits, _pTopkIdx, _pTopkW, _pShScalar,
      _pgSh, _puSh, _pprodSh, _pshDown,
      _pgExp, _puExp, _pprodExp, _pdownExp, _pwF16,
      _pExpOff, _pSorted, _pWl, _pWlc, _pTok, _tokDec,
      _pXnq, _pXnsc, _pProdq, _pProdsc,
    ]) {
      buf?.destroy();
    }
  }
}

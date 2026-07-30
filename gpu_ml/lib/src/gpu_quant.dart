import 'dart:typed_data';

import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:minigpu/minigpu.dart';

import 'gguf.dart';

/// A weight matrix (or a stack of expert matrices) held in VRAM in its
/// ORIGINAL quantized/packed encoding (GGML F16 / Q8_0 / Q4_0 / Q5_K / Q6_K)
/// as a raw u32 buffer.  Kernels unpack in registers:
///
/// - [dequantize] materializes a float32 [Tensor] (debug/interop path).
/// - [matVec] is the inference path: fused dequant dot-product against a
///   float32 activation vector, one workgroup per output row.  Weights never
///   exist in VRAM as f32 — this is the whole point for GGML models (4-8x
///   less VRAM + bandwidth than dequantize-then-matmul).
///
/// Shapes: [rows, cols] for a plain weight, or [experts, rows, cols] for a
/// MoE expert stack (GGUF ne [cols, rows, experts]).  [matVec] takes an
/// `expert:` index; the expert's byte offset is passed through a small
/// params buffer so ONE cached shader serves all experts.
///
/// Byte addressing is used inside the kernels (Q8_0 blocks are 34 bytes,
/// Q4_0 18, Q5_K 176, Q6_K 210 — none u32-aligned), so blocks straddling
/// word boundaries are handled uniformly.
class QuantizedTensor {
  QuantizedTensor._({
    required this.shape,
    required this.type,
    required this.gpu,
    required this.buffer,
  });

  /// [rows, cols] or [experts, rows, cols], outermost first.
  final List<int> shape;

  /// GGML type id (see [GgmlType]).
  final int type;

  final Minigpu gpu;

  /// Raw packed weight data, uploaded as uint32 words.
  final Buffer buffer;

  Buffer? _paramsBuffer;

  bool get isExpertStack => shape.length == 3;
  int get experts => isExpertStack ? shape[0] : 1;
  int get rows => isExpertStack ? shape[1] : shape[0];
  int get cols => isExpertStack ? shape[2] : shape[1];

  int get _bytesPerExpert {
    final traits = ggmlTypeTraits[type]!;
    return rows * cols ~/ traits.blockSize * traits.typeSize;
  }

  static const _supported = {
    GgmlType.f16,
    GgmlType.q8_0,
    GgmlType.q4_0,
    GgmlType.q5K,
    GgmlType.q6K,
  };

  /// Uploads [packedBytes] (a tensor's raw GGML-encoded data) to the GPU.
  ///
  /// [shape] is [rows, cols] or [experts, rows, cols]; blocks run along the
  /// cols (innermost) dimension, so cols must be a multiple of the type's
  /// block size (32 for Q4_0/Q8_0, 256 for K-quants, even for F16).
  static Future<QuantizedTensor> create(
    List<int> shape,
    int type,
    Uint8List packedBytes, {
    Minigpu? gpu,
  }) async {
    gpu = gpu ?? DefaultMinigpu.instance;
    if (!gpu.isInitialized) {
      await gpu.init();
    }
    if (shape.length != 2 && shape.length != 3) {
      throw Exception(
        "QuantizedTensor supports [rows, cols] or [experts, rows, cols], got $shape",
      );
    }
    if (!_supported.contains(type)) {
      throw Exception(
        "Unsupported ggml type $type (supported: f16, q8_0, q4_0, q5_k, q6_k)",
      );
    }
    final traits = ggmlTypeTraits[type]!;
    final cols = shape.last;
    if (cols % traits.blockSize != 0) {
      throw Exception(
        "cols ($cols) must be a multiple of block size ${traits.blockSize}",
      );
    }
    if (type == GgmlType.f16 && cols.isOdd) {
      throw Exception("f16 weights require even cols, got $cols");
    }
    final totalElements = shape.reduce((a, b) => a * b);
    final expectedBytes =
        (totalElements ~/ traits.blockSize) * traits.typeSize;
    if (packedBytes.length != expectedBytes) {
      throw Exception(
        "Packed data is ${packedBytes.length} bytes; shape $shape of type $type needs $expectedBytes",
      );
    }
    if (shape.length == 3 && type == GgmlType.f16) {
      final bytesPerExpert = shape[1] * shape[2] * 2;
      if (bytesPerExpert % 4 != 0) {
        throw Exception(
          "f16 expert stacks need word-aligned experts (rows*cols even)",
        );
      }
    }

    // Upload as raw bytes in bounded chunks (writeRawBytes) — no Dart-side
    // padding copy and no host allocation proportional to the tensor (weight
    // stacks run to ~1 GB each; a 40 GB model must not spike host RAM).
    // Non-word-multiple tensors (rare; small) take the legacy padded copy,
    // which also keeps the web fallback path working.
    final wordCount = (packedBytes.length + 3) ~/ 4;
    final buffer = gpu.createBuffer(wordCount * 4, BufferDataType.uint32);
    if (packedBytes.length % 4 == 0) {
      await buffer.writeRawBytes(packedBytes);
    } else {
      final words = Uint32List(wordCount);
      words.buffer.asUint8List().setRange(0, packedBytes.length, packedBytes);
      await buffer.write(words, wordCount, dataType: BufferDataType.uint32);
    }

    return QuantizedTensor._(
      shape: List.unmodifiable(shape),
      type: type,
      gpu: gpu,
      buffer: buffer,
    );
  }

  /// Streams a LARGE tensor disk→VRAM in bounded chunks: [readRange] returns
  /// the packed bytes for a range, uploaded at the same offset.  Peak host
  /// memory is one chunk (+ a small amount of driver staging) instead of the
  /// whole tensor — mandatory for multi-GB expert stacks: queued device
  /// writes accumulate in host RAM until the device is ticked, so this also
  /// flushes (a tiny readback) every few chunks.
  static Future<QuantizedTensor> createStreamed(
    List<int> shape,
    int type,
    Future<Uint8List> Function(int byteOffset, int byteLength) readRange, {
    Minigpu? gpu,
    int chunkBytes = 32 << 20,
    int flushEveryBytes = 256 << 20,
  }) async {
    gpu = gpu ?? DefaultMinigpu.instance;
    if (!gpu.isInitialized) {
      await gpu.init();
    }
    if (shape.length != 2 && shape.length != 3) {
      throw Exception(
        "QuantizedTensor supports [rows, cols] or [experts, rows, cols], got $shape",
      );
    }
    if (!_supported.contains(type)) {
      throw Exception("Unsupported ggml type $type");
    }
    final traits = ggmlTypeTraits[type]!;
    if (shape.last % traits.blockSize != 0) {
      throw Exception(
        "cols (${shape.last}) must be a multiple of block size ${traits.blockSize}",
      );
    }
    final totalElements = shape.reduce((a, b) => a * b);
    final totalBytes = (totalElements ~/ traits.blockSize) * traits.typeSize;
    assert(chunkBytes % 4 == 0);

    final wordCount = (totalBytes + 3) ~/ 4;
    final buffer = gpu.createBuffer(wordCount * 4, BufferDataType.uint32);
    final probe = Uint32List(1);
    var off = 0;
    var sinceFlush = 0;
    while (off < totalBytes) {
      var n = (totalBytes - off) < chunkBytes ? (totalBytes - off) : chunkBytes;
      final bytes = await readRange(off, n);
      final whole = n & ~3;
      if (whole > 0) {
        await buffer.writeRawBytes(Uint8List.sublistView(bytes, 0, whole),
            dstByteOffset: off);
      }
      if (whole < n) {
        // Non-word tail (last chunk only): pad those few bytes.
        final tail = Uint8List(4);
        tail.setRange(0, n - whole, Uint8List.sublistView(bytes, whole));
        await buffer.writeRawBytes(tail, dstByteOffset: off + whole);
      }
      off += n;
      sinceFlush += n;
      if (sinceFlush >= flushEveryBytes || off >= totalBytes) {
        // Tick the device so staged writes leave host RAM.
        await buffer.read(probe, 1, dataType: BufferDataType.uint32);
        sinceFlush = 0;
      }
    }

    return QuantizedTensor._(
      shape: List.unmodifiable(shape),
      type: type,
      gpu: gpu,
      buffer: buffer,
    );
  }

  void destroy() {
    _paramsBuffer?.destroy();
    _paramsBuffer = null;
    buffer.destroy();
  }

  /// Shared WGSL byte/half accessors over the packed u32 buffer `wq`.
  /// wAt() is a funnel-shift u32 load at an arbitrary EVEN byte offset (the
  /// vectorized kernels use it where block strides break word alignment).
  static const String _accessors = '''
fn byteAt(idx: u32) -> u32 {
  return (wq[idx >> 2u] >> ((idx & 3u) * 8u)) & 0xFFu;
}
fn sbyteAt(idx: u32) -> i32 {
  return bitcast<i32>(byteAt(idx) << 24u) >> 24u;
}
fn f16At(byteIdx: u32) -> f32 {
  return unpack2x16float(byteAt(byteIdx) | (byteAt(byteIdx + 1u) << 8u)).x;
}
fn wAt(byteIdx: u32) -> u32 {
  let w: u32 = byteIdx >> 2u;
  let s: u32 = (byteIdx & 3u) * 8u;
  if (s == 0u) { return wq[w]; }
  return (wq[w] >> s) | (wq[w + 1u] << (32u - s));
}
''';

  /// K-quant 6-bit scale/min unpack (ggml get_scale_min_k4).  `sb` is the
  /// byte offset of the 12-byte packed scales array.
  static const String _scaleMinK4 = '''
fn scaleMinK4(sb: u32, j: u32) -> vec2<f32> {
  var sc: u32;
  var mn: u32;
  if (j < 4u) {
    sc = byteAt(sb + j) & 63u;
    mn = byteAt(sb + j + 4u) & 63u;
  } else {
    sc = (byteAt(sb + j + 4u) & 0xFu) | ((byteAt(sb + j - 4u) >> 6u) << 4u);
    mn = (byteAt(sb + j + 4u) >> 4u) | ((byteAt(sb + j) >> 6u) << 4u);
  }
  return vec2<f32>(f32(sc), f32(mn));
}
''';

  bool get _needsScaleMinK4 => typeNeedsScaleMinK4(type);

  /// Public WGSL building blocks so fused external kernels (the decode plan)
  /// can embed the same dequant logic without duplicating it.
  static String get accessorsWGSL => _accessors;
  static String get scaleMinK4WGSL => _scaleMinK4;
  static bool typeNeedsScaleMinK4(int type) => type == GgmlType.q5K;

  /// Per-type WGSL expression assigning the dequantized value of flat
  /// element `e` (row-major over the WHOLE tensor, experts included —
  /// expert blocks are contiguous) to `v`.
  String get _dequantElementWGSL => dequantElementBodyWGSL(type);

  /// The per-element dequant body for [type]; expects `wq` and a flat
  /// element index `e` in scope, defines `v`.
  static String dequantElementBodyWGSL(int type) {
    switch (type) {
      case GgmlType.f16:
        return '''
    let pair = unpack2x16float(wq[e >> 1u]);
    let v: f32 = select(pair.x, pair.y, (e & 1u) == 1u);
''';
      case GgmlType.q8_0:
        return '''
    let blk: u32 = e / 32u;
    let l: u32 = e % 32u;
    let base: u32 = blk * 34u;
    let v: f32 = f16At(base) * f32(sbyteAt(base + 2u + l));
''';
      case GgmlType.q4_0:
        return '''
    let blk: u32 = e / 32u;
    let l: u32 = e % 32u;
    let base: u32 = blk * 18u;
    var q: i32;
    if (l < 16u) {
      q = i32(byteAt(base + 2u + l) & 0xFu) - 8;
    } else {
      q = i32(byteAt(base + 2u + (l - 16u)) >> 4u) - 8;
    }
    let v: f32 = f16At(base) * f32(q);
''';
      case GgmlType.q5K:
        return '''
    let blk: u32 = e / 256u;
    let r: u32 = e % 256u;
    let base: u32 = blk * 176u;
    let sub: u32 = r / 32u;
    let l: u32 = r % 32u;
    let grp: u32 = sub >> 1u;
    let hsel: u32 = sub & 1u;
    let sm: vec2<f32> = scaleMinK4(base + 4u, sub);
    let qlByte: u32 = byteAt(base + 48u + grp * 32u + l);
    let nib: u32 = select(qlByte & 0xFu, qlByte >> 4u, hsel == 1u);
    let hi: u32 = (byteAt(base + 16u + l) >> sub) & 1u;
    let v: f32 = f16At(base) * sm.x * f32(nib + hi * 16u) - f16At(base + 2u) * sm.y;
''';
      case GgmlType.q6K:
        return '''
    let blk: u32 = e / 256u;
    let r: u32 = e % 256u;
    let base: u32 = blk * 210u;
    let h: u32 = r / 128u;
    let rr: u32 = r % 128u;
    let quarter: u32 = rr / 32u;
    let l: u32 = rr % 32u;
    let qlb: u32 = base + h * 64u;
    let qhByte: u32 = byteAt(base + 128u + h * 32u + l);
    let scIdx: u32 = base + 192u + h * 8u + (l >> 4u) + quarter * 2u;
    var q: i32;
    if (quarter == 0u) {
      q = i32((byteAt(qlb + l) & 0xFu) | (((qhByte >> 0u) & 3u) << 4u)) - 32;
    } else if (quarter == 1u) {
      q = i32((byteAt(qlb + l + 32u) & 0xFu) | (((qhByte >> 2u) & 3u) << 4u)) - 32;
    } else if (quarter == 2u) {
      q = i32((byteAt(qlb + l) >> 4u) | (((qhByte >> 4u) & 3u) << 4u)) - 32;
    } else {
      q = i32((byteAt(qlb + l + 32u) >> 4u) | (((qhByte >> 6u) & 3u) << 4u)) - 32;
    }
    let v: f32 = f16At(base + 208u) * f32(sbyteAt(scIdx)) * f32(q);
''';
      default:
        throw Exception("Unsupported type $type");
    }
  }

  /// Materializes the full float32 tensor (all experts for a stack).
  /// Debug/interop path — inference should use [matVec].
  Future<Tensor> dequantize() async {
    final result = await Tensor.create(shape, gpu: gpu);
    final n = shape.reduce((a, b) => a * b);
    final shaderCode =
        '''
@group(0) @binding(0) var<storage, read_write> wq: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

$_accessors
${_needsScaleMinK4 ? _scaleMinK4 : ''}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let e: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (e >= ${n}u) { return; }
$_dequantElementWGSL
    output[e] = v;
}
''';
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('wq', buffer);
    shader.setBuffer('output', result.buffer);
    await shader.dispatchLinear(n);
    return result;
  }

  /// Per-type WGSL loop body accumulating this thread's partial dot product
  /// of row `row` with `x` into `acc` (256-thread strided).  `eb` is the
  /// expert byte offset (0 for 2D tensors) from the params buffer.
  String get _matVecAccumulateWGSL => matVecBodyWGSL(type);

  /// The matVec accumulate body for [type]; expects `wq`, `x`, `acc`, `row`,
  /// `lid`, `eb` and a `COLS` constant in scope.
  ///
  /// [threadVar]/[stride] parameterize the reduction width: the default
  /// (`lid.x`/`256u`) strides the whole 256-thread workgroup over one row.
  /// A narrow group (e.g. `t`/`64u`) lets several rows share a workgroup for
  /// full lane occupancy on single-token decode GEMVs, where COLS/blockSize
  /// is often < 256 and most lanes would otherwise idle.  Safe transform:
  /// the loop stride is always `+ 256u` (distinct from `/ 256u` superblock
  /// sizes and `<< Nu` shifts), and `lid.x` appears only in the loop init.
  static String matVecBodyWGSL(int type,
      {String threadVar = 'lid.x', String stride = '256u'}) {
    final body = _matVecBodyRaw(type);
    if (threadVar == 'lid.x' && stride == '256u') return body;
    return body.replaceAll('lid.x', threadVar).replaceAll('+ 256u', '+ $stride');
  }

  /// Shared-memory x staging for decode GEMVs.  On D3D11/FXC, storage-buffer
  /// reads of the activation vector inside the dot-product loop stall it
  /// ~4x (kernel-tax probe: real q8_0 body 21.8 us vs 5.2 us with x reads
  /// removed, at identical geometry/weight traffic; vec4-izing or dropping
  /// the multiplies changed nothing — the UAV x READS are the tax).  Staging
  /// x once per workgroup into a PADDED var<workgroup> array and reading via
  /// xsAt() ran 2.8x faster (7.7 us).  The +i/32 padding matters: a flat
  /// layout gives a lane-independent bank index for the q8_0 access pattern
  /// (32-way conflict — the trap that invalidated the wave-11 shared-x
  /// ablation).  Bit-exact: only the load source changes.
  ///
  /// Contract: 256-thread workgroup, `x` bound as array<vec4<f32>> (cols
  /// must be a multiple of 4), staging emitted in UNIFORM control flow
  /// before any row guard, body wrapped with [sharedXBody].
  static String xsDeclWGSL(int cols) =>
      'var<workgroup> xs: array<f32, ${cols + (cols >> 5)}>;\n'
      'fn xsAt(i: u32) -> f32 { return xs[i + (i >> 5u)]; }';

  /// The staging loop; [xBase4] offsets into x in vec4 units (per-slot
  /// windows, e.g. expert-down reading slot z's activations).
  static String stageXsWGSL(int cols, {String xBase4 = '0u'}) => '''
  for (var i4x: u32 = lid.x; i4x < ${cols ~/ 4}u; i4x = i4x + 256u) {
    let v4x: vec4<f32> = x[$xBase4 + i4x];
    let p4x: u32 = i4x * 4u + ((i4x * 4u) >> 5u);
    xs[p4x] = v4x.x; xs[p4x + 1u] = v4x.y;
    xs[p4x + 2u] = v4x.z; xs[p4x + 3u] = v4x.w;
  }
  workgroupBarrier();
''';

  /// Rewrites a matVec body's x reads to the padded shared array.  Bodies
  /// index x with window-RELATIVE expressions (no nested brackets), so the
  /// textual rewrite is safe; \b keeps idx[/xq[/xsq[ untouched.
  static String sharedXBody(String body) => body.replaceAllMapped(
      RegExp(r'\bx\[([^\]]+)\]'), (m) => 'xsAt(${m[1]})');

  static String _matVecBodyRaw(int type) {
    switch (type) {
      case GgmlType.f16:
        // Strided over u32 words = f16 pairs.  cols even + expert stacks
        // word-aligned (both enforced at create).
        return '''
  let wordsPerRow: u32 = COLS / 2u;
  let rowBase: u32 = (eb >> 2u) + row * wordsPerRow;
  for (var w: u32 = lid.x; w < wordsPerRow; w = w + 256u) {
    let pair = unpack2x16float(wq[rowBase + w]);
    acc = acc + pair.x * x[w * 2u] + pair.y * x[w * 2u + 1u];
  }
''';
      case GgmlType.q8_0:
        // Word-vectorized: one u32 load per FOUR quants instead of one per
        // byte.  Blocks are 34 bytes so `base` is always even; the quant
        // window is realigned with a funnel shift (qs is 0 or 16).  The
        // trailing `nxt` load can run one word past the tensor on the last
        // block only when qs==0, where its value is unused (robust access
        // clamps, no trap).  `unpack4xI8` sign-extends all four int8 lanes
        // in one instruction (Dawn-supported) — ~10x fewer ALU ops than the
        // four manual `bitcast<i32>(raw << Nu) >> 24u` extracts, which was
        // the decode GEMV's ALU ceiling (~330 GB/s -> memory-bound).
        return '''
  let nb: u32 = COLS / 32u;
  for (var j: u32 = lid.x; j < nb; j = j + 256u) {
    let base: u32 = eb + (row * nb + j) * 34u;
    let d: f32 = f16At(base);
    let qb: u32 = base + 2u;
    let qw: u32 = qb >> 2u;
    let qs: u32 = (qb & 3u) * 8u;
    var carry: u32 = wq[qw];
    var bsum: f32 = 0.0;
    let xb: u32 = j * 32u;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
      let nxt: u32 = wq[qw + k + 1u];
      var raw: u32;
      if (qs == 0u) { raw = carry; } else { raw = (carry >> qs) | (nxt << (32u - qs)); }
      carry = nxt;
      let xi: u32 = xb + k * 4u;
      bsum = bsum + f32(bitcast<i32>(raw << 24u) >> 24u) * x[xi]
                  + f32(bitcast<i32>(raw << 16u) >> 24u) * x[xi + 1u]
                  + f32(bitcast<i32>(raw << 8u) >> 24u) * x[xi + 2u]
                  + f32(bitcast<i32>(raw) >> 24u) * x[xi + 3u];
    }
    acc = acc + d * bsum;
  }
''';
      case GgmlType.q4_0:
        return '''
  let nb: u32 = COLS / 32u;
  for (var j: u32 = lid.x; j < nb; j = j + 256u) {
    let base: u32 = eb + (row * nb + j) * 18u;
    let d: f32 = f16At(base);
    var bsum: f32 = 0.0;
    for (var l: u32 = 0u; l < 16u; l = l + 1u) {
      let b: u32 = byteAt(base + 2u + l);
      bsum = bsum + f32(i32(b & 0xFu) - 8) * x[j * 32u + l];
      bsum = bsum + f32(i32(b >> 4u) - 8) * x[j * 32u + l + 16u];
    }
    acc = acc + d * bsum;
  }
''';
      case GgmlType.q5K:
        // Word-vectorized: 176-byte superblocks are word-aligned (eb and the
        // stride are multiples of 4), so ql/qh load as whole u32s — one load
        // per four quants.  Per-group partial sums live in vec4 lanes.
        return '''
  let nb: u32 = COLS / 256u;
  for (var j: u32 = lid.x; j < nb; j = j + 256u) {
    let base: u32 = eb + (row * nb + j) * 176u;
    let d: f32 = f16At(base);
    let dmin: f32 = f16At(base + 2u);
    let qhw0: u32 = (base + 16u) >> 2u;
    let qlw0: u32 = (base + 48u) >> 2u;
    let xb: u32 = j * 256u;
    var qLo: vec4<f32> = vec4<f32>(0.0);
    var xLo: vec4<f32> = vec4<f32>(0.0);
    var qHi: vec4<f32> = vec4<f32>(0.0);
    var xHi: vec4<f32> = vec4<f32>(0.0);
    for (var w: u32 = 0u; w < 8u; w = w + 1u) {
      let hw: u32 = wq[qhw0 + w];
      for (var g: u32 = 0u; g < 4u; g = g + 1u) {
        let lw: u32 = wq[qlw0 + g * 8u + w];
        for (var b: u32 = 0u; b < 4u; b = b + 1u) {
          let l: u32 = w * 4u + b;
          let qlB: u32 = (lw >> (b * 8u)) & 0xFFu;
          let qhB: u32 = (hw >> (b * 8u)) & 0xFFu;
          let xlo: f32 = x[xb + g * 64u + l];
          let xhi: f32 = x[xb + g * 64u + 32u + l];
          qLo[g] = qLo[g] +
              f32((qlB & 0xFu) | (((qhB >> (2u * g)) & 1u) << 4u)) * xlo;
          xLo[g] = xLo[g] + xlo;
          qHi[g] = qHi[g] +
              f32((qlB >> 4u) | (((qhB >> (2u * g + 1u)) & 1u) << 4u)) * xhi;
          xHi[g] = xHi[g] + xhi;
        }
      }
    }
    for (var g: u32 = 0u; g < 4u; g = g + 1u) {
      let smLo: vec2<f32> = scaleMinK4(base + 4u, 2u * g);
      let smHi: vec2<f32> = scaleMinK4(base + 4u, 2u * g + 1u);
      acc = acc + d * (smLo.x * qLo[g] + smHi.x * qHi[g])
                - dmin * (smLo.y * xLo[g] + smHi.y * xHi[g]);
    }
  }
''';
      case GgmlType.q6K:
        // Word-vectorized with funnel-shift loads (210-byte blocks alternate
        // between the two even alignments, so wAt() realigns each u32; the
        // one-past `nxt` load on the last block is unused or clamped).
        // Per-(quarter, scale-half) partials accumulate in vec4 lanes and
        // meet their int8 scales once per half.
        return '''
  let nb: u32 = COLS / 256u;
  for (var j: u32 = lid.x; j < nb; j = j + 256u) {
    let base: u32 = eb + (row * nb + j) * 210u;
    let d: f32 = f16At(base + 208u);
    let xb: u32 = j * 256u;
    var bsum: f32 = 0.0;
    for (var h: u32 = 0u; h < 2u; h = h + 1u) {
      let qlb: u32 = base + h * 64u;
      let qhb: u32 = base + 128u + h * 32u;
      let scb: u32 = base + 192u + h * 8u;
      let xh: u32 = xb + h * 128u;
      var qs0: vec4<f32> = vec4<f32>(0.0);
      var qs1: vec4<f32> = vec4<f32>(0.0);
      for (var w: u32 = 0u; w < 8u; w = w + 1u) {
        let lw0: u32 = wAt(qlb + w * 4u);
        let lw32: u32 = wAt(qlb + 32u + w * 4u);
        let hw: u32 = wAt(qhb + w * 4u);
        for (var b: u32 = 0u; b < 4u; b = b + 1u) {
          let l: u32 = w * 4u + b;
          let ql0: u32 = (lw0 >> (b * 8u)) & 0xFFu;
          let ql32: u32 = (lw32 >> (b * 8u)) & 0xFFu;
          let qh: u32 = (hw >> (b * 8u)) & 0xFFu;
          let v: vec4<f32> = vec4<f32>(
            f32(i32((ql0 & 0xFu) | (((qh >> 0u) & 3u) << 4u)) - 32) * x[xh + l],
            f32(i32((ql32 & 0xFu) | (((qh >> 2u) & 3u) << 4u)) - 32) * x[xh + l + 32u],
            f32(i32((ql0 >> 4u) | (((qh >> 4u) & 3u) << 4u)) - 32) * x[xh + l + 64u],
            f32(i32((ql32 >> 4u) | (((qh >> 6u) & 3u) << 4u)) - 32) * x[xh + l + 96u]);
          if (w < 4u) { qs0 = qs0 + v; } else { qs1 = qs1 + v; }
        }
      }
      for (var q: u32 = 0u; q < 4u; q = q + 1u) {
        bsum = bsum + f32(sbyteAt(scb + q * 2u)) * qs0[q]
                    + f32(sbyteAt(scb + q * 2u + 1u)) * qs1[q];
      }
    }
    acc = acc + d * bsum;
  }
''';
      default:
        throw Exception("Unsupported type $type");
    }
  }

  /// dot4I8Packed matVec body for q8_0 weights against INT8-QUANTIZED
  /// activations (per-32-block symmetric quant, matching q8_0's block size).
  /// Expects in scope: `wq` (weights), `xq: array<u32>` (packed int8
  /// activations), `xsc: array<f32>` (per-block activation scales), `acc`,
  /// `row`, `eb`, and a `COLS` const.  The int8 dot runs in ONE hardware
  /// instruction per 4 lanes (dot4I8Packed) — vs the f32 body's per-element
  /// multiply that made the GEMV arithmetic-bound (~200 vs ~600 GB/s in the
  /// bw microbench).  [xBaseWords]/[xscBaseBlocks] offset into xq/xsc for
  /// per-slot inputs (e.g. expert-down reads slot z's activations).
  static String matVecDp4aBodyWGSL(
      {String threadVar = 'lid.x',
      String stride = '256u',
      String xBaseWords = '0u',
      String xscBaseBlocks = '0u'}) {
    return '''
  let nb: u32 = COLS / 32u;
  let xwb: u32 = $xBaseWords;
  let xsb: u32 = $xscBaseBlocks;
  for (var j: u32 = $threadVar; j < nb; j = j + $stride) {
    let base: u32 = eb + (row * nb + j) * 34u;
    let dw: f32 = f16At(base);
    let qb: u32 = base + 2u;
    let qw: u32 = qb >> 2u;
    let qs: u32 = (qb & 3u) * 8u;
    var carry: u32 = wq[qw];
    var isum: i32 = 0;
    let xw: u32 = xwb + j * 8u;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
      let nxt: u32 = wq[qw + k + 1u];
      var raw: u32;
      if (qs == 0u) { raw = carry; } else { raw = (carry >> qs) | (nxt << (32u - qs)); }
      carry = nxt;
      isum = isum + dot4I8Packed(raw, xq[xw + k]);
    }
    acc = acc + dw * xsc[xsb + j] * f32(isum);
  }
''';
  }

  /// WGSL for a per-32-block int8 activation quantizer.  One workgroup per
  /// block (32 threads); expects `xin: array<f32>` (source), `xq: array<u32>`
  /// (packed int8 out), `xsc: array<f32>` (per-block scale out), and the
  /// block index from `wid`.  Symmetric (no zero point) to match q8_0.
  static const String quantizeInt8BlockWGSL = '''
var<workgroup> amax: array<f32, 32>;
var<workgroup> qsh: array<i32, 32>;

fn quantizeBlock(blk: u32, srcBase: u32, lid: u32) {
  let v: f32 = xin[srcBase + blk * 32u + lid];
  amax[lid] = abs(v);
  workgroupBarrier();
  for (var s: u32 = 16u; s > 0u; s = s >> 1u) {
    if (lid < s) { amax[lid] = max(amax[lid], amax[lid + s]); }
    workgroupBarrier();
  }
  let mx: f32 = amax[0];
  let scale: f32 = mx / 127.0;
  let inv: f32 = select(0.0, 1.0 / scale, mx > 0.0);
  qsh[lid] = clamp(i32(round(v * inv)), -127, 127);
  workgroupBarrier();
  // Lanes 0..7 each pack 4 disjoint int8 quants into one u32.
  if (lid < 8u) {
    let b0: u32 = u32(qsh[lid * 4u + 0u]) & 0xFFu;
    let b1: u32 = u32(qsh[lid * 4u + 1u]) & 0xFFu;
    let b2: u32 = u32(qsh[lid * 4u + 2u]) & 0xFFu;
    let b3: u32 = u32(qsh[lid * 4u + 3u]) & 0xFFu;
    xq[blk * 8u + lid] = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
  }
  if (lid == 0u) { xsc[blk] = scale; }
}
''';

  /// Fused dequant matrix-vector product: y = W[expert] @ x, where W is this
  /// quantized matrix (or expert stack) and [x] is a float32 vector of
  /// length cols.  Returns a float32 tensor of shape [rows].
  ///
  /// One 256-thread workgroup per row; rows fold over x/y workgroup dims
  /// past 65535.  The expert byte offset travels in a params buffer, so all
  /// experts share one cached shader.
  Future<Tensor> matVec(Tensor x, {int expert = 0}) async {
    if (x.size != cols) {
      throw Exception(
        "matVec: x has ${x.size} elements, weight cols is $cols",
      );
    }
    if (expert < 0 || expert >= experts) {
      throw Exception("expert $expert out of range (have $experts)");
    }
    final result = await Tensor.create([rows], gpu: gpu);

    _paramsBuffer ??= gpu.createBuffer(16, BufferDataType.uint32);
    final params = Uint32List(4);
    params[0] = expert * _bytesPerExpert;
    await _paramsBuffer!.write(params, 4, dataType: BufferDataType.uint32);

    final shaderCode =
        '''
@group(0) @binding(0) var<storage, read_write> wq: array<u32>;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> params: array<u32>; // [expertByteOffset]

const ROWS: u32 = ${rows}u;
const COLS: u32 = ${cols}u;

$_accessors
${_needsScaleMinK4 ? _scaleMinK4 : ''}

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
  let row: u32 = wid.x + wid.y * nwg.x;
  let eb: u32 = params[0];
  // No early return: the barriers below must be reached uniformly.
  var acc: f32 = 0.0;
  if (row < ROWS) {
$_matVecAccumulateWGSL
  }
  scratch[lid.x] = acc;
  workgroupBarrier();
  for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
    if (lid.x < s) {
      scratch[lid.x] = scratch[lid.x] + scratch[lid.x + s];
    }
    workgroupBarrier();
  }
  if (lid.x == 0u && row < ROWS) {
    y[row] = scratch[0];
  }
}
''';
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('wq', buffer);
    shader.setBuffer('x', x.buffer);
    shader.setBuffer('y', result.buffer);
    shader.setBuffer('params', _paramsBuffer!);
    final int wgX = rows <= 65535 ? rows : 65535;
    final int wgY = (rows + wgX - 1) ~/ wgX;
    await shader.dispatch(wgX, wgY, 1);
    return result;
  }
}

/// GPU loading of parsed GGUF tensors.
extension GgufGpuLoading on GgufFile {
  /// Loads a quantized/f16 weight tensor (2D, or 3D expert stack) by [name]
  /// into VRAM in its original encoding.
  Future<QuantizedTensor> loadQuantized(String name, {Minigpu? gpu}) async {
    final info = tensor(name);
    if (info == null) {
      throw Exception("GGUF tensor '$name' not found");
    }
    if (info.ne.length != 2 && info.ne.length != 3) {
      throw Exception(
        "loadQuantized supports 2D/3D tensors; '$name' has ne ${info.ne}",
      );
    }
    return QuantizedTensor.create(
      info.shape,
      info.type,
      tensorBytes(info),
      gpu: gpu,
    );
  }

  /// Loads an f32 tensor by [name] as a regular [Tensor].
  Future<Tensor> loadF32(String name, {Minigpu? gpu}) async {
    final info = tensor(name);
    if (info == null) {
      throw Exception("GGUF tensor '$name' not found");
    }
    if (info.type != GgmlType.f32) {
      throw Exception(
        "Tensor '$name' has ggml type ${info.type}, not f32 — use loadQuantized",
      );
    }
    final bytes = tensorBytes(info);
    final data = Float32List.sublistView(
      Uint8List.fromList(bytes), // copy: view alignment is not guaranteed
    );
    return Tensor.create(info.shape, data: data, gpu: gpu);
  }
}

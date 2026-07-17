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

    // Pad to a whole number of u32 words for upload.
    final wordCount = (packedBytes.length + 3) ~/ 4;
    final words = Uint32List(wordCount);
    words.buffer.asUint8List().setRange(0, packedBytes.length, packedBytes);

    final buffer = gpu.createBuffer(wordCount * 4, BufferDataType.uint32);
    await buffer.write(words, wordCount, dataType: BufferDataType.uint32);

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

  bool get _needsScaleMinK4 => type == GgmlType.q5K;

  /// Per-type WGSL expression assigning the dequantized value of flat
  /// element `e` (row-major over the WHOLE tensor, experts included —
  /// expert blocks are contiguous) to `v`.
  String get _dequantElementWGSL {
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
  String get _matVecAccumulateWGSL {
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
        return '''
  let nb: u32 = COLS / 32u;
  for (var j: u32 = lid.x; j < nb; j = j + 256u) {
    let base: u32 = eb + (row * nb + j) * 34u;
    let d: f32 = f16At(base);
    var bsum: f32 = 0.0;
    for (var l: u32 = 0u; l < 32u; l = l + 1u) {
      bsum = bsum + f32(sbyteAt(base + 2u + l)) * x[j * 32u + l];
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
        return '''
  let nb: u32 = COLS / 256u;
  for (var j: u32 = lid.x; j < nb; j = j + 256u) {
    let base: u32 = eb + (row * nb + j) * 176u;
    let d: f32 = f16At(base);
    let dmin: f32 = f16At(base + 2u);
    let xb: u32 = j * 256u;
    for (var sub: u32 = 0u; sub < 8u; sub = sub + 1u) {
      let sm: vec2<f32> = scaleMinK4(base + 4u, sub);
      let grp: u32 = sub >> 1u;
      let hsel: u32 = sub & 1u;
      var qsum: f32 = 0.0;
      var xsum: f32 = 0.0;
      for (var l: u32 = 0u; l < 32u; l = l + 1u) {
        let qlByte: u32 = byteAt(base + 48u + grp * 32u + l);
        let nib: u32 = select(qlByte & 0xFu, qlByte >> 4u, hsel == 1u);
        let hi: u32 = (byteAt(base + 16u + l) >> sub) & 1u;
        let xv: f32 = x[xb + sub * 32u + l];
        qsum = qsum + f32(nib + hi * 16u) * xv;
        xsum = xsum + xv;
      }
      acc = acc + d * sm.x * qsum - dmin * sm.y * xsum;
    }
  }
''';
      case GgmlType.q6K:
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
      for (var l: u32 = 0u; l < 32u; l = l + 1u) {
        let qhByte: u32 = byteAt(qhb + l);
        let ql0: u32 = byteAt(qlb + l);
        let ql32: u32 = byteAt(qlb + l + 32u);
        let si: u32 = l >> 4u;
        let q1: f32 = f32(i32((ql0 & 0xFu) | (((qhByte >> 0u) & 3u) << 4u)) - 32);
        let q2: f32 = f32(i32((ql32 & 0xFu) | (((qhByte >> 2u) & 3u) << 4u)) - 32);
        let q3: f32 = f32(i32((ql0 >> 4u) | (((qhByte >> 4u) & 3u) << 4u)) - 32);
        let q4: f32 = f32(i32((ql32 >> 4u) | (((qhByte >> 6u) & 3u) << 4u)) - 32);
        bsum = bsum + f32(sbyteAt(scb + si)) * q1 * x[xh + l]
                    + f32(sbyteAt(scb + si + 2u)) * q2 * x[xh + l + 32u]
                    + f32(sbyteAt(scb + si + 4u)) * q3 * x[xh + l + 64u]
                    + f32(sbyteAt(scb + si + 6u)) * q4 * x[xh + l + 96u];
      }
    }
    acc = acc + d * bsum;
  }
''';
      default:
        throw Exception("Unsupported type $type");
    }
  }

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

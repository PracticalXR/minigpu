import 'dart:convert';
import 'dart:typed_data';

/// Pure-Dart GGUF (GGML Universal File) parser — no dart:io, no GPU
/// dependency, safe for web.  Parses the header, metadata KV store, and
/// tensor directory of GGUF v2/v3 files and exposes per-tensor byte views
/// into the data section.
///
/// GPU upload of parsed tensors lives in gpu_quant.dart.

/// GGML tensor data types (the `ggml_type` enum ids used in GGUF tensor
/// directory entries).  Only a subset is loadable on GPU — see
/// [ggmlTypeTraits].
class GgmlType {
  static const int f32 = 0;
  static const int f16 = 1;
  static const int q4_0 = 2;
  static const int q4_1 = 3;
  static const int q5_0 = 6;
  static const int q5_1 = 7;
  static const int q8_0 = 8;
  static const int q8_1 = 9;
  static const int q2K = 10;
  static const int q3K = 11;
  static const int q4K = 12;
  static const int q5K = 13;
  static const int q6K = 14;
  static const int q8K = 15;
}

/// (elements per block, bytes per block) for the types this library can
/// interpret.  Q4_0: f16 scale + 32 4-bit values = 18B.  Q8_0: f16 scale +
/// 32 int8 values = 34B.  K-quants use 256-element superblocks:
/// Q5_K = d,dmin f16 + 12B packed 6-bit scales/mins + 32B high bits + 128B
/// low nibbles = 176B; Q6_K = 128B low nibbles + 64B high 2-bit bits + 16B
/// int8 scales + f16 d = 210B.
const Map<int, ({int blockSize, int typeSize})> ggmlTypeTraits = {
  GgmlType.f32: (blockSize: 1, typeSize: 4),
  GgmlType.f16: (blockSize: 1, typeSize: 2),
  GgmlType.q4_0: (blockSize: 32, typeSize: 18),
  GgmlType.q8_0: (blockSize: 32, typeSize: 34),
  GgmlType.q5K: (blockSize: 256, typeSize: 176),
  GgmlType.q6K: (blockSize: 256, typeSize: 210),
};

/// One entry in the GGUF tensor directory.
class GgufTensorInfo {
  GgufTensorInfo({
    required this.name,
    required this.ne,
    required this.type,
    required this.offset,
  });

  final String name;

  /// Dimensions as stored in GGUF: ne[0] is the INNERMOST (fastest-varying)
  /// dimension.  A llama weight of logical shape [out, in] has ne = [in, out].
  final List<int> ne;

  /// ggml_type id (see [GgmlType]).
  final int type;

  /// Byte offset of this tensor's data relative to the file's data section.
  final int offset;

  /// Total element count.
  int get elementCount => ne.fold(1, (a, b) => a * b);

  /// Shape in gpu_tensor row-major convention (outermost first) — reversed ne.
  List<int> get shape => ne.reversed.toList();

  /// Size of this tensor's data in bytes, derived from the type traits.
  /// Throws for types this library cannot size.
  int get byteSize {
    final traits = ggmlTypeTraits[type];
    if (traits == null) {
      throw Exception("Unsupported ggml type $type for tensor '$name'");
    }
    final n = elementCount;
    if (n % traits.blockSize != 0) {
      throw Exception(
        "Tensor '$name' element count $n is not a multiple of block size ${traits.blockSize}",
      );
    }
    return (n ~/ traits.blockSize) * traits.typeSize;
  }
}

/// A parsed GGUF file: metadata, tensor directory, and access to the raw
/// data section.  Operates on in-memory bytes; chunked/streaming loading is
/// a future step (UPDATE_PLAN.md Phase 4).
class GgufFile {
  GgufFile._({
    required this.version,
    required this.metadata,
    required this.tensors,
    required this.alignment,
    required this.dataOffset,
    required Uint8List bytes,
  }) : _bytes = bytes {
    for (final t in tensors) {
      _byName[t.name] = t;
    }
  }

  final int version;
  final Map<String, dynamic> metadata;
  final List<GgufTensorInfo> tensors;
  final int alignment;

  /// Absolute byte offset of the data section within the file bytes.
  final int dataOffset;

  final Uint8List _bytes;
  final Map<String, GgufTensorInfo> _byName = {};

  GgufTensorInfo? tensor(String name) => _byName[name];

  /// Raw (still quantized/packed) bytes of [info]'s data — a view, no copy.
  Uint8List tensorBytes(GgufTensorInfo info) {
    final start = dataOffset + info.offset;
    final size = info.byteSize;
    if (start + size > _bytes.length) {
      throw Exception(
        "Tensor '${info.name}' data [$start, ${start + size}) exceeds file size ${_bytes.length}",
      );
    }
    return Uint8List.sublistView(_bytes, start, start + size);
  }

  static const int _magic = 0x46554747; // "GGUF" little-endian

  static GgufFile parse(Uint8List bytes) {
    final r = _GgufReader(bytes);
    final magic = r.u32();
    if (magic != _magic) {
      throw Exception(
        "Not a GGUF file (magic 0x${magic.toRadixString(16)} != GGUF)",
      );
    }
    final version = r.u32();
    if (version < 2 || version > 3) {
      throw Exception("Unsupported GGUF version $version (supported: 2, 3)");
    }
    final tensorCount = r.u64();
    final kvCount = r.u64();

    final metadata = <String, dynamic>{};
    for (int i = 0; i < kvCount; i++) {
      final key = r.string();
      final valueType = r.u32();
      metadata[key] = r.value(valueType);
    }

    final tensors = <GgufTensorInfo>[];
    for (int i = 0; i < tensorCount; i++) {
      final name = r.string();
      final nDims = r.u32();
      if (nDims < 1 || nDims > 8) {
        throw Exception("Tensor '$name' has invalid n_dims $nDims");
      }
      final ne = List<int>.generate(nDims, (_) => r.u64());
      final type = r.u32();
      final offset = r.u64();
      tensors.add(
        GgufTensorInfo(name: name, ne: ne, type: type, offset: offset),
      );
    }

    final alignment = (metadata['general.alignment'] as int?) ?? 32;
    final dataOffset = ((r.offset + alignment - 1) ~/ alignment) * alignment;

    return GgufFile._(
      version: version,
      metadata: metadata,
      tensors: tensors,
      alignment: alignment,
      dataOffset: dataOffset,
      bytes: bytes,
    );
  }
}

/// GGUF metadata value type ids.
class _GgufValueType {
  static const int uint8 = 0;
  static const int int8 = 1;
  static const int uint16 = 2;
  static const int int16 = 3;
  static const int uint32 = 4;
  static const int int32 = 5;
  static const int float32 = 6;
  static const int boolean = 7;
  static const int string = 8;
  static const int array = 9;
  static const int uint64 = 10;
  static const int int64 = 11;
  static const int float64 = 12;
}

class _GgufReader {
  _GgufReader(this.bytes) : data = ByteData.sublistView(bytes);

  final Uint8List bytes;
  final ByteData data;
  int offset = 0;

  int u8() => data.getUint8(offset++);

  int i8() {
    final v = data.getInt8(offset);
    offset += 1;
    return v;
  }

  int u16() {
    final v = data.getUint16(offset, Endian.little);
    offset += 2;
    return v;
  }

  int i16() {
    final v = data.getInt16(offset, Endian.little);
    offset += 2;
    return v;
  }

  int u32() {
    final v = data.getUint32(offset, Endian.little);
    offset += 4;
    return v;
  }

  int i32() {
    final v = data.getInt32(offset, Endian.little);
    offset += 4;
    return v;
  }

  double f32() {
    final v = data.getFloat32(offset, Endian.little);
    offset += 4;
    return v;
  }

  double f64() {
    final v = data.getFloat64(offset, Endian.little);
    offset += 8;
    return v;
  }

  /// Web-safe u64: dart2js has no getUint64.  Values beyond 2^53 (8 PB
  /// offsets) throw rather than silently truncate.
  int u64() {
    final lo = u32();
    final hi = u32();
    if (hi > 0x1FFFFF) {
      throw Exception("u64 value exceeds JS-safe integer range");
    }
    return hi * 0x100000000 + lo;
  }

  int i64() => u64(); // Same read; practical GGUF values are non-negative.

  String string() {
    final len = u64();
    final s = utf8.decode(Uint8List.sublistView(bytes, offset, offset + len));
    offset += len;
    return s;
  }

  dynamic value(int type) {
    switch (type) {
      case _GgufValueType.uint8:
        return u8();
      case _GgufValueType.int8:
        return i8();
      case _GgufValueType.uint16:
        return u16();
      case _GgufValueType.int16:
        return i16();
      case _GgufValueType.uint32:
        return u32();
      case _GgufValueType.int32:
        return i32();
      case _GgufValueType.float32:
        return f32();
      case _GgufValueType.boolean:
        return u8() != 0;
      case _GgufValueType.string:
        return string();
      case _GgufValueType.uint64:
        return u64();
      case _GgufValueType.int64:
        return i64();
      case _GgufValueType.float64:
        return f64();
      case _GgufValueType.array:
        final elemType = u32();
        final count = u64();
        return List<dynamic>.generate(count, (_) => value(elemType));
      default:
        throw Exception("Unknown GGUF metadata value type $type");
    }
  }
}

/// Encodes [value] as IEEE 754 half-precision bits (round-to-nearest-even).
int floatToHalfBits(double value) {
  final f32 = Float32List(1)..[0] = value;
  final bits = f32.buffer.asUint32List()[0];
  final sign = (bits >> 16) & 0x8000;
  final exp = (bits >> 23) & 0xFF;
  final mant = bits & 0x7FFFFF;

  if (exp == 0xFF) {
    // Inf / NaN
    return sign | 0x7C00 | (mant != 0 ? 0x200 : 0);
  }
  // Re-bias exponent 127 -> 15.
  int e = exp - 127 + 15;
  if (e >= 0x1F) return sign | 0x7C00; // overflow -> inf
  if (e <= 0) {
    // Subnormal or zero.
    if (e < -10) return sign;
    final m = (mant | 0x800000) >> (1 - e);
    // Round half up (adequate for subnormals).
    return sign | ((m + 0x1000) >> 13);
  }
  int half = sign | (e << 10) | (mant >> 13);
  // Round to nearest even.
  final roundBit = (mant >> 12) & 1;
  final sticky = mant & 0xFFF;
  if (roundBit == 1 && (sticky != 0 || (half & 1) == 1)) {
    half += 1; // May carry into exponent — that is correct behavior.
  }
  return half;
}

/// Decodes IEEE 754 half-precision [bits] to a double.
double halfBitsToFloat(int bits) {
  final sign = (bits & 0x8000) != 0 ? -1.0 : 1.0;
  final exp = (bits >> 10) & 0x1F;
  final mant = bits & 0x3FF;
  if (exp == 0) {
    return sign * mant * 5.960464477539063e-8; // 2^-24
  }
  if (exp == 0x1F) {
    return mant == 0 ? sign * double.infinity : double.nan;
  }
  // 2^(exp - 15) * (1 + mant/1024)
  double v = 1.0 + mant / 1024.0;
  int e = exp - 15;
  while (e > 0) {
    v *= 2.0;
    e--;
  }
  while (e < 0) {
    v /= 2.0;
    e++;
  }
  return sign * v;
}

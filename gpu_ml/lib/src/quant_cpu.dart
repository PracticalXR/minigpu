import 'dart:typed_data';

import 'gguf.dart';

/// CPU dequantization of GGML-packed tensor data.  These are the reference
/// implementations the GPU kernels in gpu_quant.dart are validated against
/// (and a correctness anchor for real-file tests: GPU output must match this
/// decode of the same packed bytes exactly, while cross-quant comparisons
/// against independently encoded files catch layout misunderstandings that
/// a shared-bug reference could not).
Float32List dequantizeCpu(int type, Uint8List packed, int elementCount) {
  final traits = ggmlTypeTraits[type];
  if (traits == null) {
    throw Exception("dequantizeCpu: unsupported ggml type $type");
  }
  if (elementCount % traits.blockSize != 0) {
    throw Exception(
      "elementCount $elementCount not a multiple of block size ${traits.blockSize}",
    );
  }
  final expected = elementCount ~/ traits.blockSize * traits.typeSize;
  if (packed.length != expected) {
    throw Exception(
      "packed length ${packed.length} != expected $expected for $elementCount elements",
    );
  }
  switch (type) {
    case GgmlType.f32:
      return Float32List.sublistView(Uint8List.fromList(packed));
    case GgmlType.f16:
      return _dequantF16(packed, elementCount);
    case GgmlType.q8_0:
      return _dequantQ8_0(packed, elementCount);
    case GgmlType.q4_0:
      return _dequantQ4_0(packed, elementCount);
    case GgmlType.q5K:
      return _dequantQ5K(packed, elementCount);
    case GgmlType.q6K:
      return _dequantQ6K(packed, elementCount);
    default:
      throw Exception("dequantizeCpu: unsupported ggml type $type");
  }
}

Float32List _dequantF16(Uint8List packed, int n) {
  final bd = ByteData.sublistView(packed);
  final out = Float32List(n);
  for (int i = 0; i < n; i++) {
    out[i] = halfBitsToFloat(bd.getUint16(i * 2, Endian.little));
  }
  return out;
}

Float32List _dequantQ8_0(Uint8List packed, int n) {
  final bd = ByteData.sublistView(packed);
  final out = Float32List(n);
  for (int b = 0; b < n ~/ 32; b++) {
    final base = b * 34;
    final d = halfBitsToFloat(bd.getUint16(base, Endian.little));
    for (int l = 0; l < 32; l++) {
      out[b * 32 + l] = d * bd.getInt8(base + 2 + l);
    }
  }
  return out;
}

Float32List _dequantQ4_0(Uint8List packed, int n) {
  final bd = ByteData.sublistView(packed);
  final out = Float32List(n);
  for (int b = 0; b < n ~/ 32; b++) {
    final base = b * 18;
    final d = halfBitsToFloat(bd.getUint16(base, Endian.little));
    for (int l = 0; l < 16; l++) {
      final q = packed[base + 2 + l];
      out[b * 32 + l] = d * ((q & 0xF) - 8);
      out[b * 32 + l + 16] = d * ((q >> 4) - 8);
    }
  }
  return out;
}

/// Q5_K superblock (176B): f16 d, f16 dmin, 12B packed 6-bit scales/mins,
/// 32B qh (high bits), 128B qs (low nibbles).  8 sub-blocks of 32 elements;
/// value = d*sc*(q5) - dmin*m.
Float32List _dequantQ5K(Uint8List packed, int n) {
  final bd = ByteData.sublistView(packed);
  final out = Float32List(n);
  for (int b = 0; b < n ~/ 256; b++) {
    final base = b * 176;
    final d = halfBitsToFloat(bd.getUint16(base, Endian.little));
    final dmin = halfBitsToFloat(bd.getUint16(base + 2, Endian.little));
    for (int isb = 0; isb < 8; isb++) {
      final (sc, mn) = _scaleMinK4(packed, base + 4, isb);
      final group = isb >> 1;
      final halfSel = isb & 1;
      for (int l = 0; l < 32; l++) {
        final qlByte = packed[base + 48 + group * 32 + l];
        final nib = halfSel == 1 ? (qlByte >> 4) : (qlByte & 0xF);
        final hi = (packed[base + 16 + l] >> isb) & 1;
        out[b * 256 + isb * 32 + l] = d * sc * (nib + hi * 16) - dmin * mn;
      }
    }
  }
  return out;
}

/// The 6-bit scale/min unpack shared by K-quants (ggml get_scale_min_k4).
(int, int) _scaleMinK4(Uint8List packed, int scalesBase, int j) {
  if (j < 4) {
    return (packed[scalesBase + j] & 63, packed[scalesBase + j + 4] & 63);
  }
  final sc = (packed[scalesBase + j + 4] & 0xF) |
      ((packed[scalesBase + j - 4] >> 6) << 4);
  final mn =
      (packed[scalesBase + j + 4] >> 4) | ((packed[scalesBase + j] >> 6) << 4);
  return (sc, mn);
}

/// Q6_K superblock (210B): 128B ql (low nibbles), 64B qh (high 2 bits),
/// 16B int8 scales, f16 d.  value = d * scales[sub] * (q6 - 32).
Float32List _dequantQ6K(Uint8List packed, int n) {
  final bd = ByteData.sublistView(packed);
  final out = Float32List(n);
  for (int b = 0; b < n ~/ 256; b++) {
    final base = b * 210;
    final d = halfBitsToFloat(bd.getUint16(base + 208, Endian.little));
    for (int h = 0; h < 2; h++) {
      final qlb = base + h * 64;
      final qhb = base + 128 + h * 32;
      final scb = base + 192 + h * 8;
      final ob = b * 256 + h * 128;
      for (int l = 0; l < 32; l++) {
        final qh = packed[qhb + l];
        final ql0 = packed[qlb + l];
        final ql32 = packed[qlb + l + 32];
        final isb = l >> 4;
        final q1 = ((ql0 & 0xF) | (((qh >> 0) & 3) << 4)) - 32;
        final q2 = ((ql32 & 0xF) | (((qh >> 2) & 3) << 4)) - 32;
        final q3 = ((ql0 >> 4) | (((qh >> 4) & 3) << 4)) - 32;
        final q4 = ((ql32 >> 4) | (((qh >> 6) & 3) << 4)) - 32;
        out[ob + l] = d * bd.getInt8(scb + isb) * q1;
        out[ob + l + 32] = d * bd.getInt8(scb + isb + 2) * q2;
        out[ob + l + 64] = d * bd.getInt8(scb + isb + 4) * q3;
        out[ob + l + 96] = d * bd.getInt8(scb + isb + 6) * q4;
      }
    }
  }
  return out;
}

import 'dart:typed_data';

import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:minigpu/minigpu.dart';

/// Returns the element (as num) at [index] for a TypedData, if supported.
num getTypedDataElement(TypedData data, int index) {
  if (data is Int8List) return data[index];
  if (data is Int16List) return data[index];
  if (data is Int32List) return data[index];
  if (data is Int64List) return data[index];
  if (data is Uint8List) return data[index];
  if (data is Uint16List) return data[index];
  if (data is Uint32List) return data[index];
  if (data is Float32List) return data[index];
  if (data is Float64List) return data[index];
  throw Exception("Unsupported TypedData type: ${data.runtimeType}");
}

/// Returns a sublist of [data] from [start] (inclusive) to [end] (exclusive),
/// preserving the underlying type.
T getTypedDataSublist<T extends TypedData>(T data, int start, int end) {
  if (data is Int8List) return (data.sublist(start, end)) as T;
  if (data is Int16List) return (data.sublist(start, end)) as T;
  if (data is Int32List) return (data.sublist(start, end)) as T;
  if (data is Int64List) return (data.sublist(start, end)) as T;
  if (data is Uint8List) return (data.sublist(start, end)) as T;
  if (data is Uint16List) return (data.sublist(start, end)) as T;
  if (data is Uint32List) return (data.sublist(start, end)) as T;
  if (data is Float32List) return (data.sublist(start, end)) as T;
  if (data is Float64List) return (data.sublist(start, end)) as T;
  throw Exception("Unsupported TypedData type: ${data.runtimeType}");
}

String prepareShader(
  String template,
  BufferDataType dataType,
  Map<String, dynamic> values,
) {
  String result = template;
  values.forEach((key, value) {
    String valueString;
    if (value is int) {
      valueString = '${value}'; // Format as WGSL unsigned integer literal
    } else if (value is double) {
      // Ensure float format for WGSL f32 literal
      valueString = value.toString().contains('.')
          ? value.toString()
          : '$value.0';
      // Optionally add 'f' suffix if needed: valueString = '${valueString}f';
    } else {
      valueString = value.toString(); // Default string conversion
    }
    final wgslType = getWGSLType(dataType);
    // Replace placeholder like ${key}
    result = result.replaceAll('\${$key}', valueString);
    result = result.replaceAll('array<f32>', 'array<$wgslType>');
    result = result.replaceAll('array<i32>', 'array<$wgslType>');
    result = result.replaceAll('array<u32>', 'array<$wgslType>');
  });
  return result;
}

// Helper functions for type-conscious shader generation
String getZeroValue(BufferDataType dataType) {
  switch (dataType) {
    case BufferDataType.float32:
    case BufferDataType.float64:
    case BufferDataType.float16:
      return '0.0';
    case BufferDataType.int8:
    case BufferDataType.int16:
    case BufferDataType.int32:
    case BufferDataType.int64:
      return '0i';
    case BufferDataType.uint8:
    case BufferDataType.uint16:
    case BufferDataType.uint32:
    case BufferDataType.uint64:
      return '0u';
  }
}

String getCastExpression(String value, BufferDataType dataType) {
  switch (dataType) {
    case BufferDataType.float32:
    case BufferDataType.float64:
    case BufferDataType.float16:
      return value; // Already float
    case BufferDataType.int8:
    case BufferDataType.int16:
    case BufferDataType.int32:
    case BufferDataType.int64:
      return 'i32($value)';
    case BufferDataType.uint8:
    case BufferDataType.uint16:
    case BufferDataType.uint32:
    case BufferDataType.uint64:
      return 'u32($value)';
  }
}

String getByteConversionCode(BufferDataType dataType) {
  switch (dataType) {
    case BufferDataType.float32:
      return '''
        let byte_index = i * 4u;
        let b0 = (input_bytes[byte_index / 4u] >> ((byte_index % 4u) * 8u)) & 0xFFu;
        let b1 = (input_bytes[(byte_index + 1u) / 4u] >> (((byte_index + 1u) % 4u) * 8u)) & 0xFFu;
        let b2 = (input_bytes[(byte_index + 2u) / 4u] >> (((byte_index + 2u) % 4u) * 8u)) & 0xFFu;
        let b3 = (input_bytes[(byte_index + 3u) / 4u] >> (((byte_index + 3u) % 4u) * 8u)) & 0xFFu;
        let bits = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
        output[i] = bitcast<f32>(bits);
      ''';
    case BufferDataType.int32:
      return '''
        let byte_index = i * 4u;
        let b0 = (input_bytes[byte_index / 4u] >> ((byte_index % 4u) * 8u)) & 0xFFu;
        let b1 = (input_bytes[(byte_index + 1u) / 4u] >> (((byte_index + 1u) % 4u) * 8u)) & 0xFFu;
        let b2 = (input_bytes[(byte_index + 2u) / 4u] >> (((byte_index + 2u) % 4u) * 8u)) & 0xFFu;
        let b3 = (input_bytes[(byte_index + 3u) / 4u] >> (((byte_index + 3u) % 4u) * 8u)) & 0xFFu;
        let packed = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
        output[i] = i32(packed);
      ''';
    case BufferDataType.uint8:
      return '''
        let byte_index = i;
        let packed = (input_bytes[byte_index / 4u] >> ((byte_index % 4u) * 8u)) & 0xFFu;
        output[i] = u32(packed);
      ''';
    default:
      return 'output[i] = input_bytes[i];';
  }
}

extension TesorHelper<T extends TypedData> on Tensor<T> {
  /// _shapesCompatible checks if two shapes are compatible for broadcasting.
  bool shapesCompatible(List<int> shape1, List<int> shape2) {
    if (shape1.length != shape2.length) return false;
    for (int i = 0; i < shape1.length; i++) {
      if (shape1[i] != shape2[i] && shape1[i] != 1 && shape2[i] != 1) {
        return false;
      }
    }
    return true;
  }
}

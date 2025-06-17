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
      valueString = '${value}u'; // Format as WGSL unsigned integer literal
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

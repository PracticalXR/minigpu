import 'dart:typed_data';

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

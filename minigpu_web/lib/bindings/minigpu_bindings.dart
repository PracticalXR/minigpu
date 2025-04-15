@JS('Module')
library minigpu_bindings;

import 'dart:convert';
import 'dart:js_interop';
import 'dart:typed_data';
import 'package:js_interop_utils/js_interop_utils.dart';

typedef MGPUBuffer = JSNumber;
typedef MGPUComputeShader = JSNumber;

// js interop
@JS("HEAPU8")
external JSUint8Array HEAPU8;

@JS("HEAPU16")
external JSUint16Array HEAPU16;

@JS("HEAPU32")
external JSUint32Array HEAPU32;

@JS("HEAP8")
external JSInt8Array HEAP8;

@JS("HEAP16")
external JSInt16Array HEAP16;

@JS("HEAP32")
external JSInt32Array HEAP32;

@JS("HEAPF32")
external JSFloat32Array HEAPF32;

@JS("HEAPF64")
external JSFloat64Array HEAPF64;

Uint8List get _heapU8 => HEAPU8.toDart;
Uint16List get _heapU16 => HEAPU16.toDart;
Uint32List get _heapU32 => HEAPU32.toDart;

Int8List get _heapI8 => HEAP8.toDart;
Int16List get _heapI16 => HEAP16.toDart;
Int32List get _heapI32 => HEAP32.toDart;

Float32List get _heapF32 => HEAPF32.toDart;

Float64List get _heapF64 => HEAPF64.toDart;

@JS('_malloc')
external JSNumber _malloc(JSNumber size);

@JS('_free')
external void _free(JSNumber ptr);

Future<void> mgpuInitializeContext() async {
  await ccall(
    "mgpuInitializeContext".toJS,
    "void".toJS,
    <JSAny>[].toJSDeep,
    <JSAny>[].toJSDeep,
    {"async": true}.toJSDeep,
  ).toDart;
}

@JS('_mgpuDestroyContext')
external void _mgpuDestroyContext();

void mgpuDestroyContext() {
  _mgpuDestroyContext();
}

// Compute shader functions
@JS('_mgpuCreateComputeShader')
external MGPUComputeShader _mgpuCreateComputeShader();

MGPUComputeShader mgpuCreateComputeShader() {
  MGPUComputeShader shader = _mgpuCreateComputeShader();
  return shader;
}

@JS('_mgpuDestroyComputeShader')
external void _mgpuDestroyComputeShader(MGPUComputeShader shader);

void mgpuDestroyComputeShader(MGPUComputeShader shader) {
  _mgpuDestroyComputeShader(shader);
}

@JS('allocateUTF8')
external JSString allocateUTF8(String str);

@JS('_mgpuLoadKernel')
external void _mgpuLoadKernel(MGPUComputeShader shader, JSNumber kernelString);

void mgpuLoadKernel(MGPUComputeShader shader, String kernelString) {
  final bytes = utf8.encode(kernelString);
  final kernelBytes = Uint8List(bytes.length + 1)
    ..setRange(0, bytes.length, bytes)
    ..[bytes.length] = 0; // null terminator

  // Allocate memory for the string.
  final allocSize = kernelBytes.length * kernelBytes.elementSizeInBytes;
  final ptr = _malloc(allocSize.toJS);
  try {
    _heapU8.setAll(ptr.toDartInt, kernelBytes);
    _mgpuLoadKernel(shader, ptr);
  } finally {
    _free(ptr);
  }
}

@JS('_mgpuHasKernel')
external JSBoolean _mgpuHasKernel(MGPUComputeShader shader);

bool mgpuHasKernel(MGPUComputeShader shader) {
  return _mgpuHasKernel(shader).dartify() as bool;
}

// Buffer functions
@JS('_mgpuCreateBuffer')
external MGPUBuffer _mgpuCreateBuffer(JSNumber bufferSize);

MGPUBuffer mgpuCreateBuffer(int bufferSize) {
  return _mgpuCreateBuffer(bufferSize.toJS);
}

@JS('_mgpuDestroyBuffer')
external void _mgpuDestroyBuffer(MGPUBuffer buffer);

void mgpuDestroyBuffer(MGPUBuffer buffer) {
  _mgpuDestroyBuffer(buffer);
}

@JS('_mgpuSetBuffer')
external void _mgpuSetBuffer(
  MGPUComputeShader shader,
  JSNumber tag,
  MGPUBuffer buffer,
);

void mgpuSetBuffer(MGPUComputeShader shader, int tag, MGPUBuffer buffer) {
  try {
    _mgpuSetBuffer(shader, tag.toJS, buffer);
  } finally {}
}

Future<void> mgpuDispatch(
  MGPUComputeShader shader,
  int groupsX,
  int groupsY,
  int groupsZ,
) async {
  try {
    await ccall(
      "mgpuDispatch".toJS,
      "void".toJS,
      ["number", "number", "number", "number", "number"].toJSDeep,
      [shader, groupsX.toJS, groupsY.toJS, groupsZ.toJS].toJSDeep,
      {"async": true}.toJSDeep,
    ).toDart;
  } finally {}
}

@JS('ccall')
external JSPromise ccall(
  JSString name,
  JSString returnType,
  JSArray argTypes,
  JSArray args,
  JSObject opts,
);

/// Asynchronous read functions for multiple types.
/// Each function allocates native memory, calls ccall with async:true,
/// then copies the data from the WebAssembly heap into the given Dart TypedData.

Future<void> mgpuReadBufferAsyncInt8(
  MGPUBuffer buffer,
  Int8List outputData, {
  int readElements = 0,
  int elementOffset = 0,
  int readBytes = 0,
  int byteOffset = 0,
}) async {
  final int bytesPerElem = Int8List.bytesPerElement;
  final int sizeToRead = (readElements > 0)
      ? readElements * bytesPerElem
      : (readBytes > 0
          ? readBytes
          : (outputData.length - elementOffset) * bytesPerElem);
  final int effectiveByteOffset =
      (readElements > 0) ? elementOffset * bytesPerElem : byteOffset;
  final JSNumber ptr = _malloc(sizeToRead.toJS);
  final int startIndex = ptr.toDartInt ~/ bytesPerElem;
  try {
    await ccall(
      "mgpuReadBufferAsyncInt8".toJS,
      "number".toJS,
      ["number", "number", "number", "number"].toJSDeep,
      [buffer, ptr, sizeToRead.toJS, effectiveByteOffset.toJS].toJSDeep,
      {"async": true}.toJSDeep,
    ).toDart;
    final int elementsToRead = sizeToRead ~/ bytesPerElem;
    final output = _heapI8.sublist(startIndex, startIndex + elementsToRead);
    outputData.setAll(elementOffset, output);
  } finally {
    _free(ptr);
  }
}

Future<void> mgpuReadBufferAsyncInt16(
  MGPUBuffer buffer,
  Int16List outputData, {
  int readElements = 0,
  int elementOffset = 0,
  int readBytes = 0,
  int byteOffset = 0,
}) async {
  final int bytesPerElem = Int16List.bytesPerElement;
  final int sizeToRead = (readElements > 0)
      ? readElements * bytesPerElem
      : (readBytes > 0
          ? readBytes
          : (outputData.length - elementOffset) * bytesPerElem);
  final int effectiveByteOffset =
      (readElements > 0) ? elementOffset * bytesPerElem : byteOffset;
  final JSNumber ptr = _malloc(sizeToRead.toJS);
  final int startIndex = ptr.toDartInt ~/ bytesPerElem;
  try {
    await ccall(
      "mgpuReadBufferAsyncInt16".toJS,
      "number".toJS,
      ["number", "number", "number", "number"].toJSDeep,
      [buffer, ptr, sizeToRead.toJS, effectiveByteOffset.toJS].toJSDeep,
      {"async": true}.toJSDeep,
    ).toDart;
    final int elementsToRead = sizeToRead ~/ bytesPerElem;
    final heapInt16 = _heapI16.buffer.asInt16List();
    final output = heapInt16.sublist(startIndex, startIndex + elementsToRead);
    outputData.setAll(elementOffset, output);
  } finally {
    _free(ptr);
  }
}

Future<void> mgpuReadBufferAsyncInt32(
  MGPUBuffer buffer,
  Int32List outputData, {
  int readElements = 0,
  int elementOffset = 0,
  int readBytes = 0,
  int byteOffset = 0,
}) async {
  final int bytesPerElem = Int32List.bytesPerElement;
  final int sizeToRead = (readElements > 0)
      ? readElements * bytesPerElem
      : (readBytes > 0
          ? readBytes
          : (outputData.length - elementOffset) * bytesPerElem);
  final int effectiveByteOffset =
      (readElements > 0) ? elementOffset * bytesPerElem : byteOffset;
  final JSNumber ptr = _malloc(sizeToRead.toJS);
  final int startIndex = ptr.toDartInt ~/ bytesPerElem;
  try {
    await ccall(
      "mgpuReadBufferAsyncInt32".toJS,
      "number".toJS,
      ["number", "number", "number", "number"].toJSDeep,
      [buffer, ptr, sizeToRead.toJS, effectiveByteOffset.toJS].toJSDeep,
      {"async": true}.toJSDeep,
    ).toDart;
    final int elementsToRead = sizeToRead ~/ bytesPerElem;
    final heapInt32 = _heapI32.buffer.asInt32List();
    final output = heapInt32.sublist(startIndex, startIndex + elementsToRead);
    outputData.setAll(elementOffset, output);
  } finally {
    _free(ptr);
  }
}

Future<void> mgpuReadBufferAsyncInt64(
  MGPUBuffer buffer,
  ByteData outputData, {
  int readElements = 0,
  int elementOffset = 0,
  int readBytes = 0,
  int byteOffset = 0,
}) async {
  throw UnimplementedError(
      'mgpuReadBufferAsyncInt64 not implemented in this version');
  // Use 8 bytes per element.
  final int bytesPerElem = 8;
  final int sizeToRead = (readElements > 0)
      ? readElements * bytesPerElem
      : (readBytes > 0
          ? readBytes
          : (outputData.lengthInBytes - elementOffset * bytesPerElem));
  final int effectiveByteOffset =
      (readElements > 0) ? elementOffset * bytesPerElem : byteOffset;
  final JSNumber ptr = _malloc(sizeToRead.toJS);
  final int startIndex = ptr.toDartInt ~/ bytesPerElem;
  try {
    await ccall(
      "mgpuReadBufferAsyncInt64".toJS,
      "number".toJS,
      ["number", "number", "number", "number"].toJSDeep,
      [buffer, ptr, sizeToRead.toJS, effectiveByteOffset.toJS].toJSDeep,
      {"async": true}.toJSDeep,
    ).toDart;
    final int elementsToRead = sizeToRead ~/ bytesPerElem;
    final ByteData heapBD = _heapU8.buffer.asByteData();
    for (int i = 0; i < elementsToRead; i++) {
      final int value = heapBD.getInt64(
          startIndex * bytesPerElem + i * bytesPerElem, Endian.little);
      outputData.setInt64(elementOffset + i, value, Endian.little);
    }
  } finally {
    _free(ptr);
  }
}

Future<void> mgpuReadBufferAsyncUint8(
  MGPUBuffer buffer,
  Uint8List outputData, {
  int readElements = 0,
  int elementOffset = 0,
  int readBytes = 0,
  int byteOffset = 0,
}) async {
  final int bytesPerElem = Uint8List.bytesPerElement;
  final int sizeToRead = (readElements > 0)
      ? readElements * bytesPerElem
      : (readBytes > 0
          ? readBytes
          : (outputData.length - elementOffset) * bytesPerElem);
  final int effectiveByteOffset =
      (readElements > 0) ? elementOffset * bytesPerElem : byteOffset;
  final JSNumber ptr = _malloc(sizeToRead.toJS);
  final int startIndex = ptr.toDartInt ~/ bytesPerElem;
  try {
    await ccall(
      "mgpuReadBufferAsyncUint8".toJS,
      "number".toJS,
      ["number", "number", "number", "number"].toJSDeep,
      [buffer, ptr, sizeToRead.toJS, effectiveByteOffset.toJS].toJSDeep,
      {"async": true}.toJSDeep,
    ).toDart;
    final int elementsToRead = sizeToRead ~/ bytesPerElem;
    final output = _heapU8.sublist(startIndex, startIndex + elementsToRead);
    outputData.setAll(elementOffset, output);
  } finally {
    _free(ptr);
  }
}

Future<void> mgpuReadBufferAsyncUint16(
  MGPUBuffer buffer,
  Uint16List outputData, {
  int readElements = 0,
  int elementOffset = 0,
  int readBytes = 0,
  int byteOffset = 0,
}) async {
  final int bytesPerElem = Uint16List.bytesPerElement;
  final int sizeToRead = (readElements > 0)
      ? readElements * bytesPerElem
      : (readBytes > 0
          ? readBytes
          : (outputData.length - elementOffset) * bytesPerElem);
  final int effectiveByteOffset =
      (readElements > 0) ? elementOffset * bytesPerElem : byteOffset;
  final JSNumber ptr = _malloc(sizeToRead.toJS);
  final int startIndex = ptr.toDartInt ~/ bytesPerElem;
  try {
    await ccall(
      "mgpuReadBufferAsyncUint16".toJS,
      "number".toJS,
      ["number", "number", "number", "number"].toJSDeep,
      [buffer, ptr, sizeToRead.toJS, effectiveByteOffset.toJS].toJSDeep,
      {"async": true}.toJSDeep,
    ).toDart;
    final int elementsToRead = sizeToRead ~/ bytesPerElem;
    final output = _heapU16.sublist(startIndex, startIndex + elementsToRead);
    outputData.setAll(elementOffset, output);
  } finally {
    _free(ptr);
  }
}

Future<void> mgpuReadBufferAsyncUint32(
  MGPUBuffer buffer,
  Uint32List outputData, {
  int readElements = 0,
  int elementOffset = 0,
  int readBytes = 0,
  int byteOffset = 0,
}) async {
  final int bytesPerElem = Uint32List.bytesPerElement;
  final int sizeToRead = (readElements > 0)
      ? readElements * bytesPerElem
      : (readBytes > 0
          ? readBytes
          : (outputData.length - elementOffset) * bytesPerElem);
  final int effectiveByteOffset =
      (readElements > 0) ? elementOffset * bytesPerElem : byteOffset;
  final JSNumber ptr = _malloc(sizeToRead.toJS);
  final int startIndex = ptr.toDartInt ~/ bytesPerElem;
  try {
    await ccall(
      "mgpuReadBufferAsyncUint32".toJS,
      "number".toJS,
      ["number", "number", "number", "number"].toJSDeep,
      [buffer, ptr, sizeToRead.toJS, effectiveByteOffset.toJS].toJSDeep,
      {"async": true}.toJSDeep,
    ).toDart;
    final int elementsToRead = sizeToRead ~/ bytesPerElem;
    final output = _heapU32.sublist(startIndex, startIndex + elementsToRead);
    outputData.setAll(elementOffset, output);
  } finally {
    _free(ptr);
  }
}

Future<void> mgpuReadBufferAsyncUint64(
  MGPUBuffer buffer,
  Uint64List outputData, {
  int readElements = 0,
  int elementOffset = 0,
  int readBytes = 0,
  int byteOffset = 0,
}) async {
  throw UnimplementedError(
      'mgpuReadBufferAsyncUint64 not implemented in this version');
  final int bytesPerElem = Uint64List.bytesPerElement;
  final int sizeToRead = (readElements > 0)
      ? readElements * bytesPerElem
      : (readBytes > 0
          ? readBytes
          : (outputData.length - elementOffset) * bytesPerElem);
  final int effectiveByteOffset =
      (readElements > 0) ? elementOffset * bytesPerElem : byteOffset;
  final JSNumber ptr = _malloc(sizeToRead.toJS);
  final int startIndex = ptr.toDartInt ~/ bytesPerElem;
  try {
    await ccall(
      "mgpuReadBufferAsyncUint64".toJS,
      "number".toJS,
      ["number", "number", "number", "number"].toJSDeep,
      [buffer, ptr, sizeToRead.toJS, effectiveByteOffset.toJS].toJSDeep,
      {"async": true}.toJSDeep,
    ).toDart;
    final int elementsToRead = sizeToRead ~/ bytesPerElem;
    final Uint64List heapUint64 = _heapU8.buffer.asUint64List();
    final output = heapUint64.sublist(startIndex, startIndex + elementsToRead);
    outputData.setAll(elementOffset, output);
  } finally {
    _free(ptr);
  }
}

Future<void> mgpuReadBufferAsyncFloat(
  MGPUBuffer buffer,
  Float32List outputData, {
  int readElements = 0,
  int elementOffset = 0,
  int readBytes = 0,
  int byteOffset = 0,
}) async {
  final int bytesPerElem = Float32List.bytesPerElement;
  final int sizeToRead = (readElements > 0)
      ? readElements * bytesPerElem
      : (readBytes > 0
          ? readBytes
          : (outputData.length - elementOffset) * bytesPerElem);
  final int effectiveByteOffset =
      (readElements > 0) ? elementOffset * bytesPerElem : byteOffset;
  final JSNumber ptr = _malloc(sizeToRead.toJS);
  final int startIndex = ptr.toDartInt ~/ bytesPerElem;
  try {
    await ccall(
      "mgpuReadBufferSync".toJS,
      "number".toJS,
      ["number", "number", "number", "number"].toJSDeep,
      [buffer, ptr, sizeToRead.toJS, effectiveByteOffset.toJS].toJSDeep,
      {"async": true}.toJSDeep,
    ).toDart;
    final int elementsToRead = sizeToRead ~/ bytesPerElem;
    final output = _heapF32.sublist(startIndex, startIndex + elementsToRead);
    outputData.setAll(elementOffset, output);
  } finally {
    _free(ptr);
  }
}

Future<void> mgpuReadBufferAsyncDouble(
  MGPUBuffer buffer,
  Float64List outputData, {
  int readElements = 0,
  int elementOffset = 0,
  int readBytes = 0,
  int byteOffset = 0,
}) async {
  final int bytesPerElem = Float64List.bytesPerElement;
  final int sizeToRead = (readElements > 0)
      ? readElements * bytesPerElem
      : (readBytes > 0
          ? readBytes
          : (outputData.length - elementOffset) * bytesPerElem);
  final int effectiveByteOffset =
      (readElements > 0) ? elementOffset * bytesPerElem : byteOffset;
  final JSNumber ptr = _malloc(sizeToRead.toJS);
  final int startIndex = ptr.toDartInt ~/ bytesPerElem;
  try {
    await ccall(
      "mgpuReadBufferSync".toJS,
      "number".toJS,
      ["number", "number", "number", "number"].toJSDeep,
      [buffer, ptr, sizeToRead.toJS, effectiveByteOffset.toJS].toJSDeep,
      {"async": true}.toJSDeep,
    ).toDart;
    final int elementsToRead = sizeToRead ~/ bytesPerElem;
    final output = _heapF64.sublist(startIndex, startIndex + elementsToRead);
    outputData.setAll(elementOffset, output);
  } finally {
    _free(ptr);
  }
}

/// New setBufferData functions for additional types.
/// Each function allocates WASM memory, copies the Dart TypedData into the HEAP,
/// then calls the native setter.

@JS('_mgpuSetBufferDataInt8')
external void _mgpuSetBufferDataInt8(
    MGPUBuffer buffer, JSNumber inputDataPtr, JSNumber size);

void mgpuSetBufferDataInt8(MGPUBuffer buffer, Int8List inputData, int size) {
  final int byteSize = size * Int8List.bytesPerElement;
  final JSNumber ptr = _malloc(byteSize.toJS);
  final int startIndex = ptr.toDartInt;
  try {
    _heapU8.setRange(startIndex, startIndex + inputData.length, inputData);
    _mgpuSetBufferDataInt8(buffer, ptr, byteSize.toJS);
  } finally {
    _free(ptr);
  }
}

@JS('_mgpuSetBufferDataInt16')
external void _mgpuSetBufferDataInt16(
    MGPUBuffer buffer, JSNumber inputDataPtr, JSNumber size);

void mgpuSetBufferDataInt16(MGPUBuffer buffer, Int16List inputData, int size) {
  final int byteSize = size * Int16List.bytesPerElement;
  final JSNumber ptr = _malloc(byteSize.toJS);
  final int startIndex = ptr.toDartInt;
  try {
    final heapInt16 = _heapU8.buffer.asInt16List();
    heapInt16.setRange(
        startIndex ~/ 2, (startIndex ~/ 2) + inputData.length, inputData);
    _mgpuSetBufferDataInt16(buffer, ptr, byteSize.toJS);
  } finally {
    _free(ptr);
  }
}

@JS('_mgpuSetBufferDataInt32')
external void _mgpuSetBufferDataInt32(
    MGPUBuffer buffer, JSNumber inputDataPtr, JSNumber size);

void mgpuSetBufferDataInt32(MGPUBuffer buffer, Int32List inputData, int size) {
  final int byteSize = size * Int32List.bytesPerElement;
  final JSNumber ptr = _malloc(byteSize.toJS);
  final int startIndex = ptr.toDartInt ~/ Int32List.bytesPerElement;
  try {
    final heapInt32 = _heapU8.buffer.asInt32List();
    heapInt32.setRange(startIndex, startIndex + inputData.length, inputData);
    _mgpuSetBufferDataInt32(buffer, ptr, byteSize.toJS);
  } finally {
    _free(ptr);
  }
}

@JS('_mgpuSetBufferDataInt64')
external void _mgpuSetBufferDataInt64(
    MGPUBuffer buffer, JSNumber inputDataPtr, JSNumber size);

void mgpuSetBufferDataInt64(MGPUBuffer buffer, Int64List inputData, int size) {
  final int byteSize = size * Int64List.bytesPerElement;
  final JSNumber ptr = _malloc(byteSize.toJS);
  final int startIndex = ptr.toDartInt ~/ Int64List.bytesPerElement;
  try {
    final heapInt64 = _heapU8.buffer.asInt64List();
    heapInt64.setRange(startIndex, startIndex + inputData.length, inputData);
    _mgpuSetBufferDataInt64(buffer, ptr, byteSize.toJS);
  } finally {
    _free(ptr);
  }
}

@JS('_mgpuSetBufferDataUint8')
external void _mgpuSetBufferDataUint8(
    MGPUBuffer buffer, JSNumber inputDataPtr, JSNumber size);

void mgpuSetBufferDataUint8(MGPUBuffer buffer, Uint8List inputData, int size) {
  final int byteSize = size * Uint8List.bytesPerElement;
  final JSNumber ptr = _malloc(byteSize.toJS);
  final int startIndex = ptr.toDartInt;
  try {
    _heapU8.setRange(startIndex, startIndex + inputData.length, inputData);
    _mgpuSetBufferDataUint8(buffer, ptr, byteSize.toJS);
  } finally {
    _free(ptr);
  }
}

@JS('_mgpuSetBufferDataUint16')
external void _mgpuSetBufferDataUint16(
    MGPUBuffer buffer, JSNumber inputDataPtr, JSNumber size);

void mgpuSetBufferDataUint16(
    MGPUBuffer buffer, Uint16List inputData, int size) {
  final int byteSize = size * Uint16List.bytesPerElement;
  final JSNumber ptr = _malloc(byteSize.toJS);
  final int startIndex = ptr.toDartInt ~/ Uint16List.bytesPerElement;
  try {
    final heapUint16 = _heapU8.buffer.asUint16List();
    heapUint16.setRange(startIndex, startIndex + inputData.length, inputData);
    _mgpuSetBufferDataUint16(buffer, ptr, byteSize.toJS);
  } finally {
    _free(ptr);
  }
}

@JS('_mgpuSetBufferDataUint32')
external void _mgpuSetBufferDataUint32(
    MGPUBuffer buffer, JSNumber inputDataPtr, JSNumber size);

void mgpuSetBufferDataUint32(
    MGPUBuffer buffer, Uint32List inputData, int size) {
  final int byteSize = size * Uint32List.bytesPerElement;
  final JSNumber ptr = _malloc(byteSize.toJS);
  final int startIndex = ptr.toDartInt ~/ Uint32List.bytesPerElement;
  try {
    final heapUint32 = _heapU8.buffer.asUint32List();
    heapUint32.setRange(startIndex, startIndex + inputData.length, inputData);
    _mgpuSetBufferDataUint32(buffer, ptr, byteSize.toJS);
  } finally {
    _free(ptr);
  }
}

@JS('_mgpuSetBufferDataUint64')
external void _mgpuSetBufferDataUint64(
    MGPUBuffer buffer, JSNumber inputDataPtr, JSNumber size);

void mgpuSetBufferDataUint64(
    MGPUBuffer buffer, Uint64List inputData, int size) {
  final int byteSize = size * Uint64List.bytesPerElement;
  final JSNumber ptr = _malloc(byteSize.toJS);
  final int startIndex = ptr.toDartInt ~/ Uint64List.bytesPerElement;
  try {
    final heapUint64 = _heapU8.buffer.asUint64List();
    heapUint64.setRange(startIndex, startIndex + inputData.length, inputData);
    _mgpuSetBufferDataUint64(buffer, ptr, byteSize.toJS);
  } finally {
    _free(ptr);
  }
}

@JS('_mgpuSetBufferDataFloat')
external void _mgpuSetBufferDataFloat(
    MGPUBuffer buffer, JSNumber inputDataPtr, JSNumber size);

void mgpuSetBufferDataFloat(
    MGPUBuffer buffer, Float32List inputData, int size) {
  final int byteSize = size * Float32List.bytesPerElement;
  final JSNumber ptr = _malloc(byteSize.toJS);
  final int startIndex = ptr.toDartInt ~/ Float32List.bytesPerElement;
  try {
    final heapFloat32 = _heapF32.buffer.asFloat32List();
    heapFloat32.setRange(startIndex, startIndex + inputData.length, inputData);
    _mgpuSetBufferDataFloat(buffer, ptr, byteSize.toJS);
  } finally {
    _free(ptr);
  }
}

@JS('_mgpuSetBufferDataDouble')
external void _mgpuSetBufferDataDouble(
    MGPUBuffer buffer, JSNumber inputDataPtr, JSNumber size);

void mgpuSetBufferDataDouble(
    MGPUBuffer buffer, Float64List inputData, int size) {
  final int byteSize = size * Float64List.bytesPerElement;
  final JSNumber ptr = _malloc(byteSize.toJS);
  // For doubles, we assume HEAPU8's buffer can be viewed as Float64List.
  final int startIndex = ptr.toDartInt ~/ Float64List.bytesPerElement;
  try {
    final heapFloat64 = _heapU8.buffer.asFloat64List();
    heapFloat64.setRange(startIndex, startIndex + inputData.length, inputData);
    _mgpuSetBufferDataDouble(buffer, ptr, byteSize.toJS);
  } finally {
    _free(ptr);
  }
}

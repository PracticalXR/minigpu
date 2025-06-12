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

Future<void> mgpuDestroyContext() async {
  await ccall(
    "mgpuDestroyContext".toJS,
    "void".toJS,
    <JSAny>[].toJSDeep,
    <JSAny>[].toJSDeep,
    {"async": true}.toJSDeep,
  ).toDart;
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

  final allocSize = kernelBytes.length;
  final ptr = _malloc(allocSize.toJS);
  final int startIndex = ptr.toDartInt;

  // Bounds check
  if (startIndex < 0 || startIndex + kernelBytes.length > _heapU8.length) {
    _free(ptr);
    throw StateError(
      'Kernel string allocation would exceed heap bounds: '
      'trying to write ${kernelBytes.length} bytes at index $startIndex '
      'but heap size is ${_heapU8.length}',
    );
  }

  _heapU8.setRange(startIndex, startIndex + kernelBytes.length, kernelBytes);
  _mgpuLoadKernel(shader, ptr);

  _free(ptr);
}

@JS('_mgpuHasKernel')
external JSBoolean _mgpuHasKernel(MGPUComputeShader shader);

bool mgpuHasKernel(MGPUComputeShader shader) {
  return _mgpuHasKernel(shader).dartify() as bool;
}

// Buffer functions
@JS('_mgpuCreateBuffer')
external MGPUBuffer _mgpuCreateBuffer(JSNumber bufferSize, JSNumber dataType);

MGPUBuffer mgpuCreateBuffer(int elements, int dataType) {
  return _mgpuCreateBuffer(elements.toJS, dataType.toJS);
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

Future<void> mgpuReadAsyncInt8(
  MGPUBuffer buffer,
  Int8List outputData, {
  int readElements = 0,
  int elementOffset = 0,
}) async {
  final int bytesPerElem = Int8List.bytesPerElement;
  // Calculate element count based on outputData size if readElements is 0
  final int elementsToRead = (readElements > 0)
      ? readElements
      : (outputData.length - elementOffset);
  final int sizeToAllocate =
      elementsToRead * bytesPerElem; // Allocate based on elements

  if (elementsToRead <= 0 ||
      elementOffset < 0 ||
      elementOffset >= outputData.length)
    return; // Nothing to read or invalid offset

  final JSNumber ptr = _malloc(sizeToAllocate.toJS);
  final int startIndex = ptr.toDartInt; // Byte index for heap copy

  try {
    // Add bounds check
    if (startIndex < 0 || startIndex + sizeToAllocate > _heapU8.length) {
      throw StateError(
        'Int8 buffer read would exceed heap bounds: '
        'trying to read $elementsToRead elements at index $startIndex '
        'but heap size is ${_heapU8.length}',
      );
    }
    await ccall(
      "mgpuReadSyncInt8".toJS, // Use sync C++ function
      "void".toJS, // C++ function returns void
      ["number", "number", "number", "number"].toJSDeep,
      [buffer, ptr, elementsToRead.toJS, elementOffset.toJS].toJSDeep,
      {"async": true}.toJSDeep,
    ).toDart;

    // Copy the data from WASM heap
    // startIndex is the byte offset, elementsToRead is the count
    final output = _heapI8.sublist(startIndex, startIndex + elementsToRead);
    outputData.setAll(elementOffset, output);
  } finally {
    _free(ptr);
  }
}

Future<void> mgpuReadAsyncInt16(
  MGPUBuffer buffer,
  Int16List outputData, {
  int readElements = 0,
  int elementOffset = 0,
}) async {
  final int bytesPerElem = Int16List.bytesPerElement;
  final int elementsToRead = (readElements > 0)
      ? readElements
      : (outputData.length - elementOffset);
  final int sizeToAllocate = elementsToRead * bytesPerElem;

  if (elementsToRead <= 0 ||
      elementOffset < 0 ||
      elementOffset >= outputData.length)
    return;

  final JSNumber ptr = _malloc(sizeToAllocate.toJS);
  // Element index for heap copy
  final int startIndex = ptr.toDartInt ~/ bytesPerElem;

  try {
    // Add bounds check for the heap access
    if (startIndex < 0 || startIndex + elementsToRead > _heapI16.length) {
      throw StateError(
        'Int16 buffer read would exceed heap bounds: '
        'trying to read $elementsToRead elements at index $startIndex '
        'but heap size is ${_heapI16.length}',
      );
    }
    await ccall(
      "mgpuReadSyncInt16".toJS, // Use sync C++ function
      "void".toJS,
      ["number", "number", "number", "number"].toJSDeep,
      [buffer, ptr, elementsToRead.toJS, elementOffset.toJS].toJSDeep,
      {"async": true}.toJSDeep,
    ).toDart;

    // Use Int16 view for copying
    final output = _heapI16.sublist(startIndex, startIndex + elementsToRead);
    outputData.setAll(elementOffset, output);
  } finally {
    _free(ptr);
  }
}

Future<void> mgpuReadAsyncInt32(
  MGPUBuffer buffer,
  Int32List outputData, {
  int readElements = 0,
  int elementOffset = 0,
}) async {
  final int bytesPerElem = Int32List.bytesPerElement;
  final int elementsToRead = (readElements > 0)
      ? readElements
      : (outputData.length - elementOffset);
  final int sizeToAllocate = elementsToRead * bytesPerElem;

  if (elementsToRead <= 0 ||
      elementOffset < 0 ||
      elementOffset >= outputData.length)
    return;

  final JSNumber ptr = _malloc(sizeToAllocate.toJS);
  final int startIndex = ptr.toDartInt ~/ bytesPerElem;

  try {
    // Add bounds check for the heap access
    if (startIndex < 0 || startIndex + elementsToRead > _heapI32.length) {
      throw StateError(
        'Int32 buffer read would exceed heap bounds: '
        'trying to read $elementsToRead elements at index $startIndex '
        'but heap size is ${_heapI32.length}',
      );
    }
    await ccall(
      "mgpuReadSyncInt32".toJS, // Use sync C++ function
      "void".toJS,
      ["number", "number", "number", "number"].toJSDeep,
      [buffer, ptr, elementsToRead.toJS, elementOffset.toJS].toJSDeep,
      {"async": true}.toJSDeep,
    ).toDart;

    final output = _heapI32.sublist(startIndex, startIndex + elementsToRead);
    outputData.setAll(elementOffset, output);
  } finally {
    _free(ptr);
  }
}

Future<void> mgpuReadAsyncInt64(
  MGPUBuffer buffer,
  // Note: Dart web doesn't have Int64List directly, use ByteData or handle BigInt conversion
  // Assuming caller provides appropriate ByteData or handles conversion from Uint8List view
  TypedData outputData, {
  int readElements = 0,
  int elementOffset = 0,
}) async {
  final int bytesPerElem = 8; // Int64
  final int elementsToRead = (readElements > 0)
      ? readElements
      : ((outputData.lengthInBytes ~/ bytesPerElem) - elementOffset);
  final int sizeToAllocate = elementsToRead * bytesPerElem;

  if (elementsToRead <= 0 ||
      elementOffset < 0 ||
      (elementOffset * bytesPerElem) >= outputData.lengthInBytes)
    return;

  final JSNumber ptr = _malloc(sizeToAllocate.toJS);
  // Byte index for heap copy
  final int startByteIndex = ptr.toDartInt;

  try {
    await ccall(
      "mgpuReadSyncInt64".toJS, // Use sync C++ function
      "void".toJS,
      ["number", "number", "number", "number"].toJSDeep,
      [buffer, ptr, elementsToRead.toJS, elementOffset.toJS].toJSDeep,
      {"async": true}.toJSDeep,
    ).toDart;

    final heapBytes = _heapU8.sublist(
      startByteIndex,
      startByteIndex + sizeToAllocate,
    );

    if (outputData is ByteData) {
      // Ensure we don't write past the end of the ByteData buffer
      final int bytesAvailableInOut =
          outputData.lengthInBytes - (elementOffset * bytesPerElem);
      final int bytesToCopy = sizeToAllocate <= bytesAvailableInOut
          ? sizeToAllocate
          : bytesAvailableInOut;
      if (bytesToCopy > 0) {
        final outputBytes = Uint8List.view(
          outputData.buffer,
          outputData.offsetInBytes + (elementOffset * bytesPerElem),
          bytesToCopy,
        );
        outputBytes.setRange(0, bytesToCopy, heapBytes.sublist(0, bytesToCopy));
      }
    } else if (outputData is Uint8List) {
      // Allow direct copy if caller uses Uint8List
      final int bytesAvailableInOut =
          outputData.lengthInBytes - (elementOffset * bytesPerElem);
      final int bytesToCopy = sizeToAllocate <= bytesAvailableInOut
          ? sizeToAllocate
          : bytesAvailableInOut;
      if (bytesToCopy > 0) {
        outputData.setRange(
          elementOffset * bytesPerElem,
          (elementOffset * bytesPerElem) + bytesToCopy,
          heapBytes.sublist(0, bytesToCopy),
        );
      }
    } else {
      // Consider throwing an error for unsupported outputData types for Int64
      print(
        "Warning: mgpuReadAsyncInt64 expects outputData to be ByteData or Uint8List",
      );
    }
  } finally {
    _free(ptr);
  }
}

Future<void> mgpuReadAsyncUint8(
  MGPUBuffer buffer,
  Uint8List outputData, {
  int readElements = 0,
  int elementOffset = 0,
}) async {
  final int bytesPerElem = Uint8List.bytesPerElement;
  final int elementsToRead = (readElements > 0)
      ? readElements
      : (outputData.length - elementOffset);
  final int sizeToAllocate = elementsToRead * bytesPerElem;

  if (elementsToRead <= 0 ||
      elementOffset < 0 ||
      elementOffset >= outputData.length)
    return;

  final JSNumber ptr = _malloc(sizeToAllocate.toJS);
  final int startIndex = ptr.toDartInt; // Byte index

  try {
    // Add bounds check for the heap access
    if (startIndex < 0 || startIndex + sizeToAllocate > _heapU8.length) {
      throw StateError(
        'Uint8 buffer read would exceed heap bounds: '
        'trying to read $elementsToRead elements at index $startIndex '
        'but heap size is ${_heapU8.length}',
      );
    }
    await ccall(
      "mgpuReadSyncUint8".toJS, // Use sync C++ function
      "void".toJS,
      ["number", "number", "number", "number"].toJSDeep,
      [buffer, ptr, elementsToRead.toJS, elementOffset.toJS].toJSDeep,
      {"async": true}.toJSDeep,
    ).toDart;

    final output = _heapU8.sublist(startIndex, startIndex + elementsToRead);
    outputData.setAll(elementOffset, output);
  } finally {
    _free(ptr);
  }
}

Future<void> mgpuReadAsyncUint16(
  MGPUBuffer buffer,
  Uint16List outputData, {
  int readElements = 0,
  int elementOffset = 0,
}) async {
  final int bytesPerElem = Uint16List.bytesPerElement;
  final int elementsToRead = (readElements > 0)
      ? readElements
      : (outputData.length - elementOffset);
  final int sizeToAllocate = elementsToRead * bytesPerElem;

  if (elementsToRead <= 0 ||
      elementOffset < 0 ||
      elementOffset >= outputData.length)
    return;

  final JSNumber ptr = _malloc(sizeToAllocate.toJS);
  final int startIndex = ptr.toDartInt ~/ bytesPerElem;

  try {
    // Add bounds check for the heap access
    if (startIndex < 0 || startIndex + elementsToRead > _heapU16.length) {
      throw StateError(
        'Uint16 buffer read would exceed heap bounds: '
        'trying to read $elementsToRead elements at index $startIndex '
        'but heap size is ${_heapU16.length}',
      );
    }
    await ccall(
      "mgpuReadSyncUint16".toJS, // Use sync C++ function
      "void".toJS,
      ["number", "number", "number", "number"].toJSDeep,
      [buffer, ptr, elementsToRead.toJS, elementOffset.toJS].toJSDeep,
      {"async": true}.toJSDeep,
    ).toDart;

    final output = _heapU16.sublist(startIndex, startIndex + elementsToRead);
    outputData.setAll(elementOffset, output);
  } finally {
    _free(ptr);
  }
}

Future<void> mgpuReadAsyncUint32(
  MGPUBuffer buffer,
  Uint32List outputData, {
  int readElements = 0,
  int elementOffset = 0,
}) async {
  final int bytesPerElem = Uint32List.bytesPerElement;
  final int elementsToRead = (readElements > 0)
      ? readElements
      : (outputData.length - elementOffset);
  final int sizeToAllocate = elementsToRead * bytesPerElem;

  if (elementsToRead <= 0 ||
      elementOffset < 0 ||
      elementOffset >= outputData.length)
    return;

  final JSNumber ptr = _malloc(sizeToAllocate.toJS);
  final int startIndex = ptr.toDartInt ~/ bytesPerElem;

  try {
    // Add bounds check for the heap access
    if (startIndex < 0 || startIndex + elementsToRead > _heapU32.length) {
      throw StateError(
        'Uint32 buffer read would exceed heap bounds: '
        'trying to read $elementsToRead elements at index $startIndex '
        'but heap size is ${_heapU32.length}',
      );
    }
    await ccall(
      "mgpuReadSyncUint32".toJS, // Use sync C++ function
      "void".toJS,
      ["number", "number", "number", "number"].toJSDeep,
      [buffer, ptr, elementsToRead.toJS, elementOffset.toJS].toJSDeep,
      {"async": true}.toJSDeep,
    ).toDart;

    final output = _heapU32.sublist(startIndex, startIndex + elementsToRead);
    outputData.setAll(elementOffset, output);
  } finally {
    _free(ptr);
  }
}

Future<void> mgpuReadAsyncUint64(
  MGPUBuffer buffer,
  // Note: Dart web doesn't have Uint64List directly, use ByteData or handle BigInt conversion
  // Assuming caller provides appropriate ByteData or handles conversion from Uint8List view
  TypedData outputData, {
  int readElements = 0,
  int elementOffset = 0,
}) async {
  final int bytesPerElem = 8; // Uint64
  final int elementsToRead = (readElements > 0)
      ? readElements
      : ((outputData.lengthInBytes ~/ bytesPerElem) - elementOffset);
  final int sizeToAllocate = elementsToRead * bytesPerElem;

  if (elementsToRead <= 0 ||
      elementOffset < 0 ||
      (elementOffset * bytesPerElem) >= outputData.lengthInBytes)
    return;

  final JSNumber ptr = _malloc(sizeToAllocate.toJS);
  final int startByteIndex = ptr.toDartInt; // Byte index

  try {
    // Add bounds check for the heap access
    if (startByteIndex < 0 ||
        startByteIndex + sizeToAllocate > _heapU8.length) {
      throw StateError(
        'Uint64 buffer read would exceed heap bounds: '
        'trying to read $elementsToRead elements at index $startByteIndex '
        'but heap size is ${_heapU8.length}',
      );
    }
    await ccall(
      "mgpuReadSyncUint64".toJS, // Use sync C++ function
      "void".toJS,
      ["number", "number", "number", "number"].toJSDeep,
      [buffer, ptr, elementsToRead.toJS, elementOffset.toJS].toJSDeep,
      {"async": true}.toJSDeep,
    ).toDart;

    // Copy using Uint8List view as Uint64List is not standard in dart:html
    final heapBytes = _heapU8.sublist(
      startByteIndex,
      startByteIndex + sizeToAllocate,
    );

    if (outputData is ByteData) {
      final int bytesAvailableInOut =
          outputData.lengthInBytes - (elementOffset * bytesPerElem);
      final int bytesToCopy = sizeToAllocate <= bytesAvailableInOut
          ? sizeToAllocate
          : bytesAvailableInOut;
      if (bytesToCopy > 0) {
        final outputBytes = Uint8List.view(
          outputData.buffer,
          outputData.offsetInBytes + (elementOffset * bytesPerElem),
          bytesToCopy,
        );
        outputBytes.setRange(0, bytesToCopy, heapBytes.sublist(0, bytesToCopy));
      }
    } else if (outputData is Uint8List) {
      // Allow direct copy if caller uses Uint8List
      final int bytesAvailableInOut =
          outputData.lengthInBytes - (elementOffset * bytesPerElem);
      final int bytesToCopy = sizeToAllocate <= bytesAvailableInOut
          ? sizeToAllocate
          : bytesAvailableInOut;
      if (bytesToCopy > 0) {
        outputData.setRange(
          elementOffset * bytesPerElem,
          (elementOffset * bytesPerElem) + bytesToCopy,
          heapBytes.sublist(0, bytesToCopy),
        );
      }
    } else {
      // Consider throwing an error for unsupported outputData types for Uint64
      print(
        "Warning: mgpuReadAsyncUint64 expects outputData to be ByteData or Uint8List",
      );
    }
  } finally {
    _free(ptr);
  }
}

Future<void> mgpuReadAsyncFloat(
  MGPUBuffer buffer,
  Float32List outputData, {
  int readElements = 0,
  int elementOffset = 0,
}) async {
  final int bytesPerElem = Float32List.bytesPerElement;
  final int elementsToRead = (readElements > 0)
      ? readElements
      : (outputData.length - elementOffset);
  final int sizeToAllocate = elementsToRead * bytesPerElem;

  if (elementsToRead <= 0 ||
      elementOffset < 0 ||
      elementOffset >= outputData.length)
    return;

  final JSNumber ptr = _malloc((sizeToAllocate + 64).toJS);
  final int startByteIndex = ptr.toDartInt; //  This is correct - byte index
  final int startElementIndex =
      startByteIndex ~/ bytesPerElem; //  Convert to element index

  try {
    // Add bounds check for the heap access using ELEMENT indices
    if (startElementIndex < 0 ||
        startElementIndex + elementsToRead > _heapF32.length) {
      throw StateError(
        'Float32 buffer read would exceed heap bounds: '
        'trying to read $elementsToRead elements at element index $startElementIndex '
        'but heap size is ${_heapF32.length}',
      );
    }

    await ccall(
      "mgpuReadSyncFloat32".toJS,
      "void".toJS,
      ["number", "number", "number", "number"].toJSDeep,
      [buffer, ptr, elementsToRead.toJS, elementOffset.toJS].toJSDeep,
      {"async": true}.toJSDeep,
    ).toDart;

    // Use the ELEMENT index for the typed array access
    final output = _heapF32.sublist(
      startElementIndex,
      startElementIndex + elementsToRead,
    );

    outputData.setAll(elementOffset, output);
  } finally {
    _free(ptr);
  }
}

Future<void> mgpuReadAsyncDouble(
  MGPUBuffer buffer,
  Float64List outputData, {
  int readElements = 0,
  int elementOffset = 0,
}) async {
  final int bytesPerElem = Float64List.bytesPerElement;
  final int elementsToRead = (readElements > 0)
      ? readElements
      : (outputData.length - elementOffset);
  final int sizeToAllocate = elementsToRead * bytesPerElem;

  if (elementsToRead <= 0 ||
      elementOffset < 0 ||
      elementOffset >= outputData.length)
    return;

  final JSNumber ptr = _malloc(sizeToAllocate.toJS);
  final int startIndex = ptr.toDartInt ~/ bytesPerElem;

  try {
    // Add bounds check
    if (startIndex < 0 || startIndex + elementsToRead > _heapF64.length) {
      throw StateError(
        'Float64 buffer read would exceed heap bounds: '
        'trying to read $elementsToRead elements at index $startIndex '
        'but heap size is ${_heapF64.length}',
      );
    }
    // Assuming C++ function is named mgpuReadSyncFloat64 now
    await ccall(
      "mgpuReadSyncFloat64".toJS, // Use specific sync C++ function
      "void".toJS,
      ["number", "number", "number", "number"].toJSDeep,
      [buffer, ptr, elementsToRead.toJS, elementOffset.toJS].toJSDeep,
      {"async": true}.toJSDeep,
    ).toDart;

    final output = _heapF64.sublist(startIndex, startIndex + elementsToRead);
    outputData.setAll(elementOffset, output);
  } finally {
    _free(ptr);
  }
}

@JS('_mgpuWriteInt8')
external void _mgpuWriteInt8(
  MGPUBuffer buffer,
  JSNumber inputDataPtr,
  JSNumber size,
);

void mgpuWriteInt8(MGPUBuffer buffer, Int8List inputData, int size) {
  final int elementsToWrite = size > 0 ? size : inputData.length;
  final int byteSize = elementsToWrite * Int8List.bytesPerElement;
  final JSNumber ptr = _malloc(byteSize.toJS);
  final int startIndex = ptr.toDartInt;
  try {
    final int actualElements = elementsToWrite <= inputData.length
        ? elementsToWrite
        : inputData.length;
    _heapU8.setRange(
      startIndex,
      startIndex + actualElements,
      inputData.sublist(0, actualElements),
    );
    _mgpuWriteInt8(buffer, ptr, byteSize.toJS);
  } finally {
    _free(ptr);
  }
}

@JS('_mgpuWriteInt16')
external void _mgpuWriteInt16(
  MGPUBuffer buffer,
  JSNumber inputDataPtr,
  JSNumber size,
);

void mgpuWriteInt16(MGPUBuffer buffer, Int16List inputData, int size) {
  final int elementsToWrite = size > 0 ? size : inputData.length;
  final int byteSize = elementsToWrite * Int16List.bytesPerElement;
  final JSNumber ptr = _malloc(byteSize.toJS);
  final int startByteIndex = ptr.toDartInt;
  try {
    final int startElementIndex = startByteIndex ~/ Int16List.bytesPerElement;
    final int actualElements = elementsToWrite <= inputData.length
        ? elementsToWrite
        : inputData.length;

    if (startElementIndex + actualElements > _heapI16.length) {
      throw StateError('Int16 buffer allocation would exceed heap bounds');
    }

    _heapI16.setRange(
      startElementIndex,
      startElementIndex + actualElements,
      inputData.sublist(0, actualElements),
    );
    _mgpuWriteInt16(buffer, ptr, byteSize.toJS);
  } finally {
    _free(ptr);
  }
}

@JS('_mgpuWriteInt32')
external void _mgpuWriteInt32(
  MGPUBuffer buffer,
  JSNumber inputDataPtr,
  JSNumber size,
);

void mgpuWriteInt32(MGPUBuffer buffer, Int32List inputData, int size) {
  final int elementsToWrite = size > 0 ? size : inputData.length;
  final int byteSize = elementsToWrite * Int32List.bytesPerElement;
  final JSNumber ptr = _malloc(byteSize.toJS);
  final int startByteIndex = ptr.toDartInt;
  try {
    final int startElementIndex = startByteIndex ~/ Int32List.bytesPerElement;
    final int actualElements = elementsToWrite <= inputData.length
        ? elementsToWrite
        : inputData.length;

    if (startElementIndex + actualElements > _heapI32.length) {
      throw StateError('Int32 buffer allocation would exceed heap bounds');
    }

    _heapI32.setRange(
      startElementIndex,
      startElementIndex + actualElements,
      inputData.sublist(0, actualElements),
    );
    _mgpuWriteInt32(buffer, ptr, byteSize.toJS);
  } finally {
    _free(ptr);
  }
}

@JS('_mgpuWriteInt64')
external void _mgpuWriteInt64(
  MGPUBuffer buffer,
  JSNumber inputDataPtr,
  JSNumber size,
);

void mgpuWriteInt64(MGPUBuffer buffer, Int64List inputData, int size) {
  final int byteSize = size * Int64List.bytesPerElement;
  final JSNumber ptr = _malloc(byteSize.toJS);
  final int startIndex = ptr.toDartInt ~/ Int64List.bytesPerElement;
  try {
    final heapInt64 = _heapU8.buffer.asInt64List();
    heapInt64.setRange(startIndex, startIndex + inputData.length, inputData);
    _mgpuWriteInt64(buffer, ptr, byteSize.toJS);
  } finally {
    _free(ptr);
  }
}

@JS('_mgpuWriteUint8')
external void _mgpuWriteUint8(
  MGPUBuffer buffer,
  JSNumber inputDataPtr,
  JSNumber size,
);

void mgpuWriteUint8(MGPUBuffer buffer, Uint8List inputData, int size) {
  final int elementsToWrite = size > 0 ? size : inputData.length;
  final int byteSize = elementsToWrite * Uint8List.bytesPerElement;
  final JSNumber ptr = _malloc(byteSize.toJS);
  final int startIndex = ptr.toDartInt;
  try {
    final int actualElements = elementsToWrite <= inputData.length
        ? elementsToWrite
        : inputData.length;
    _heapU8.setRange(
      startIndex,
      startIndex + actualElements,
      inputData.sublist(0, actualElements),
    );
    _mgpuWriteUint8(buffer, ptr, byteSize.toJS);
  } finally {
    _free(ptr);
  }
}

@JS('_mgpuWriteUint16')
external void _mgpuWriteUint16(
  MGPUBuffer buffer,
  JSNumber inputDataPtr,
  JSNumber size,
);

void mgpuWriteUint16(MGPUBuffer buffer, Uint16List inputData, int size) {
  final int byteSize = size * Uint16List.bytesPerElement;
  final JSNumber ptr = _malloc(byteSize.toJS);
  final int startByteIndex = ptr.toDartInt;
  try {
    final int startElementIndex = startByteIndex ~/ Uint16List.bytesPerElement;

    if (startElementIndex + inputData.length > _heapU16.length) {
      throw StateError('Uint16 buffer allocation would exceed heap bounds');
    }

    _heapU16.setRange(
      startElementIndex,
      startElementIndex + inputData.length,
      inputData,
    );
    _mgpuWriteUint16(buffer, ptr, byteSize.toJS);
  } finally {
    _free(ptr);
  }
}

@JS('_mgpuWriteUint32')
external void _mgpuWriteUint32(
  MGPUBuffer buffer,
  JSNumber inputDataPtr,
  JSNumber size,
);

void mgpuWriteUint32(MGPUBuffer buffer, Uint32List inputData, int size) {
  final int elementsToWrite = size > 0 ? size : inputData.length;
  final int byteSize = elementsToWrite * Uint32List.bytesPerElement;
  final JSNumber ptr = _malloc(byteSize.toJS);
  final int startByteIndex = ptr.toDartInt;
  try {
    final int startElementIndex = startByteIndex ~/ Uint32List.bytesPerElement;
    final int actualElements = elementsToWrite <= inputData.length
        ? elementsToWrite
        : inputData.length;

    if (startElementIndex + actualElements > _heapU32.length) {
      throw StateError('Uint32 buffer allocation would exceed heap bounds');
    }

    _heapU32.setRange(
      startElementIndex,
      startElementIndex + actualElements,
      inputData.sublist(0, actualElements),
    );
    _mgpuWriteUint32(buffer, ptr, byteSize.toJS);
  } finally {
    _free(ptr);
  }
}

@JS('_mgpuWriteUint64')
external void _mgpuWriteUint64(
  MGPUBuffer buffer,
  JSNumber inputDataPtr,
  JSNumber size,
);

void mgpuWriteUint64(MGPUBuffer buffer, Uint64List inputData, int size) {
  final int byteSize = size * Uint64List.bytesPerElement;
  final JSNumber ptr = _malloc(byteSize.toJS);
  final int startIndex = ptr.toDartInt ~/ Uint64List.bytesPerElement;
  try {
    final heapUint64 = _heapU8.buffer.asUint64List();
    heapUint64.setRange(startIndex, startIndex + inputData.length, inputData);
    _mgpuWriteUint64(buffer, ptr, byteSize.toJS);
  } finally {
    _free(ptr);
  }
}

Future<void> mgpuWriteFloat(
  MGPUBuffer buffer,
  Float32List inputData,
  int size,
) async {
  final int elementsToWrite = size > 0 ? size : inputData.length;
  final int byteSize = elementsToWrite * Float32List.bytesPerElement;

  final JSNumber ptr = _malloc(byteSize.toJS);
  final int startIndex = ptr.toDartInt ~/ Float32List.bytesPerElement;

  try {
    final int actualElements = elementsToWrite <= inputData.length
        ? elementsToWrite
        : inputData.length;

    final currentHeapF32 = HEAPF32.toDart;

    if (startIndex < 0 || startIndex + actualElements > currentHeapF32.length) {
      throw StateError(
        'Float32 buffer allocation would exceed heap bounds: '
        'trying to write $actualElements elements at index $startIndex '
        'but heap size is ${currentHeapF32.length}',
      );
    }

    currentHeapF32.setRange(
      startIndex,
      startIndex + actualElements,
      inputData.sublist(0, actualElements),
    );

    // Make this async using ccall
    await ccall(
      "mgpuWriteFloat".toJS,
      "void".toJS,
      ["number", "number", "number"].toJSDeep,
      [buffer, ptr, byteSize.toJS].toJSDeep,
      {"async": true}.toJSDeep,
    ).toDart;
  } finally {
    _free(ptr);
  }
}

@JS('_mgpuWriteDouble')
external void _mgpuWriteDouble(
  MGPUBuffer buffer,
  JSNumber inputDataPtr,
  JSNumber size,
);

void mgpuWriteDouble(MGPUBuffer buffer, Float64List inputData, int size) {
  final int elementsToWrite = size > 0 ? size : inputData.length;
  final int byteSize = elementsToWrite * Float64List.bytesPerElement;
  final JSNumber ptr = _malloc(byteSize.toJS);
  final int startByteIndex = ptr.toDartInt;
  try {
    final int startElementIndex = startByteIndex ~/ Float64List.bytesPerElement;
    final int actualElements = elementsToWrite <= inputData.length
        ? elementsToWrite
        : inputData.length;

    if (startElementIndex + actualElements > _heapF64.length) {
      throw StateError('Float64 buffer allocation would exceed heap bounds');
    }

    _heapF64.setRange(
      startElementIndex,
      startElementIndex + actualElements,
      inputData.sublist(0, actualElements),
    );
    _mgpuWriteDouble(buffer, ptr, byteSize.toJS);
  } finally {
    _free(ptr);
  }
}

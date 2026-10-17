// 🔴 Scoped to `MinigpuModule`, NOT `Module`: classic Emscripten builds all
// claim `globalThis.Module`, and any other wasm on the page (miniav_web's
// audio module in the meet app) clobbers it — every `_mgpu*` extern then
// resolves on the WRONG module and dies as "func is not a function".
// `module_loader.dart` publishes the glue's private instance under this name
// (or adopts a host-page-preloaded global `Module`); nothing here may read
// the shared global.
@JS('MinigpuModule')
library minigpu_bindings;

import 'dart:async';
import 'dart:convert';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';
import 'package:js_interop_utils/js_interop_utils.dart';

import 'module_loader.dart';
import 'webgpu_interop.dart';

export 'module_loader.dart'
    show ensureMinigpuModuleLoaded, isMinigpuModuleLoaded;

typedef MGPUBuffer = JSNumber;
typedef MGPUComputeShader = JSNumber;
typedef MGPUTexture = JSNumber;
typedef MGPUTextureView = JSNumber;
typedef MGPUSampler = JSNumber;
typedef MGPURenderPipeline = JSNumber;

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

// 🔴 -sASYNCIFY keeps ONE unwound stack and ONE promise slot per module: a
// second suspending ccall entered while another is asleep overwrites both, so
// the first caller never resumes and its unwind block leaks (release builds
// carry no assert). Every call that can suspend — the reads and context
// init/destroy — enters through this FIFO, never from inside another entry.
// An idle queue starts the call at once.
int _suspendingCalls = 0;
Future<void> _suspendingTail = Future<void>.value();

Future<T> _serialAsync<T>(Future<T> Function() call) {
  final previous = _suspendingTail;
  final done = Completer<void>();
  _suspendingTail = done.future;
  final result = _suspendingCalls++ == 0
      ? Future<T>.sync(call)
      : previous.then((_) => call());
  void settle() {
    _suspendingCalls--;
    done.complete();
  }

  // Registered before the caller's own await, so the count has dropped by
  // the time the caller resumes (mgpuDestroyBuffer reads it).
  result.then<void>((_) => settle(), onError: (Object _) => settle());
  return result;
}

Future<void> mgpuInitializeContext() async {
  // Everything else in this library assumes an initialized context, so this
  // is the one choke point where the module itself must be up first.
  await ensureMinigpuModuleLoaded();
  await _serialAsync<void>(() async {
    await ccall(
      "mgpuInitializeContext".toJS,
      "void".toJS,
      <JSAny>[].toJS,
      <JSAny>[].toJS,
      {"async": true}.toJSDeep,
    ).toDart;
  });
}

Future<void> mgpuDestroyContext() async {
  await _serialAsync<void>(() async {
    await ccall(
      "mgpuDestroyContext".toJS,
      "void".toJS,
      <JSAny>[].toJS,
      <JSAny>[].toJS,
      {"async": true}.toJSDeep,
    ).toDart;
  });
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

  HEAPU8.set(Uint8ListToJSUint8Array(kernelBytes).toJS, startIndex);
  _mgpuLoadKernel(shader, ptr);
  if (!mgpuHasKernel(shader)) {
    _free(ptr);
    throw StateError('Kernel not accepted by native layer');
  }
  _free(ptr);
}

@JS('_mgpuHasKernel')
external JSNumber _mgpuHasKernel(MGPUComputeShader shader);

bool mgpuHasKernel(MGPUComputeShader shader) {
  return _mgpuHasKernel(shader).toDartInt == 1 ? true : false;
}

// Buffer functions
@JS('_mgpuCreateBuffer')
external MGPUBuffer _mgpuCreateBuffer(JSNumber bufferSize, JSNumber dataType);

MGPUBuffer mgpuCreateBuffer(int elements, int dataType) {
  return _mgpuCreateBuffer(elements.toJS, dataType.toJS);
}

bool _hasMinigpuExport(String name) {
  final module = globalContext.getProperty<JSObject?>('MinigpuModule'.toJS);
  if (module == null || module.isUndefinedOrNull) {
    return false;
  }
  final value = module.getProperty<JSAny?>(name.toJS);
  return value.typeofEquals('function');
}

@JS('_mgpuBufferDescriptorVersion')
external JSNumber _mgpuBufferDescriptorVersion();

@JS('_mgpuCreateBufferWithUsage')
external MGPUBuffer _mgpuCreateBufferWithUsage(
  JSNumber sizeBytes,
  JSNumber usage,
);

bool get hasBufferDescriptorV1 =>
    _hasMinigpuExport('_mgpuBufferDescriptorVersion') &&
    _hasMinigpuExport('_mgpuCreateBufferWithUsage') &&
    _mgpuBufferDescriptorVersion().toDartInt == 1;

MGPUBuffer? mgpuCreateBufferWithUsage(int sizeBytes, int usage) {
  if (!hasBufferDescriptorV1) return null;
  final result = _mgpuCreateBufferWithUsage(sizeBytes.toJS, usage.toJS);
  return result.toDartInt == 0 ? null : result;
}

@JS('_mgpuBufferBindingVersion')
external JSNumber _mgpuBufferBindingVersion();

@JS('_mgpuSetBufferRange')
external JSNumber _mgpuSetBufferRange(
  MGPUComputeShader shader,
  JSNumber slot,
  MGPUBuffer buffer,
  JSNumber kind,
  JSNumber offsetBytes,
  JSNumber sizeBytes,
);

bool get hasBufferBindingV1 =>
    _hasMinigpuExport('_mgpuBufferBindingVersion') &&
    _hasMinigpuExport('_mgpuSetBufferRange') &&
    _mgpuBufferBindingVersion().toDartInt == 1;

int mgpuSetBufferRange(
  MGPUComputeShader shader,
  int slot,
  MGPUBuffer buffer,
  int kind,
  int offsetBytes,
  int sizeBytes,
) => _mgpuSetBufferRange(
  shader,
  slot.toJS,
  buffer,
  kind.toJS,
  offsetBytes.toJS,
  sizeBytes.toJS,
).toDartInt;

@JS('_mgpuTextureResourceVersion')
external JSNumber _mgpuTextureResourceVersion();

@JS('_mgpuCreateTexture2D')
external MGPUTexture _mgpuCreateTexture2D(
  JSNumber width,
  JSNumber height,
  JSNumber format,
  JSNumber usage,
  JSNumber layers,
  JSNumber mipLevels,
);

@JS('_mgpuCreateTextureView')
external MGPUTextureView _mgpuCreateTextureView(
  MGPUTexture texture,
  JSNumber dimension,
  JSNumber baseMip,
  JSNumber mipCount,
  JSNumber baseLayer,
  JSNumber layerCount,
);

@JS('_mgpuCreateSampler')
external MGPUSampler _mgpuCreateSampler(
  JSNumber addressU,
  JSNumber addressV,
  JSNumber addressW,
  JSNumber mag,
  JSNumber min,
  JSNumber mip,
  JSNumber lodMin,
  JSNumber lodMax,
  JSNumber compare,
);

@JS('_mgpuWriteTexture2D')
external JSNumber _mgpuWriteTexture2D(
  MGPUTexture texture,
  JSNumber bytes,
  JSNumber byteCount,
  JSNumber mipLevel,
  JSNumber arrayLayer,
);

@JS('_mgpuTextureRegionUploadVersion')
external JSNumber _mgpuTextureRegionUploadVersion();

@JS('_mgpuWriteTextureRegion2D')
external JSNumber _mgpuWriteTextureRegion2D(
  MGPUTexture texture,
  JSNumber bytes,
  JSNumber byteCount,
  JSNumber x,
  JSNumber y,
  JSNumber width,
  JSNumber height,
  JSNumber mipLevel,
  JSNumber arrayLayer,
);

bool get hasTextureRegionUploadV1 =>
    _hasMinigpuExport('_mgpuTextureRegionUploadVersion') &&
    _hasMinigpuExport('_mgpuWriteTextureRegion2D') &&
    _mgpuTextureRegionUploadVersion().toDartInt == 1;

@JS('_mgpuSetOwnedTexture')
external JSNumber _mgpuSetOwnedTexture(
  MGPUComputeShader shader,
  JSNumber slot,
  MGPUTextureView view,
  JSNumber kind,
);

@JS('_mgpuSetOwnedSampler')
external JSNumber _mgpuSetOwnedSampler(
  MGPUComputeShader shader,
  JSNumber slot,
  MGPUSampler sampler,
  JSNumber kind,
);

@JS('_mgpuDestroyTexture')
external void _mgpuDestroyTexture(MGPUTexture texture);

@JS('_mgpuDestroyTextureView')
external void _mgpuDestroyTextureView(MGPUTextureView view);

@JS('_mgpuDestroySampler')
external void _mgpuDestroySampler(MGPUSampler sampler);

bool get hasTextureResourceV1 =>
    _hasMinigpuExport('_mgpuTextureResourceVersion') &&
    _hasMinigpuExport('_mgpuCreateTexture2D') &&
    _hasMinigpuExport('_mgpuCreateTextureView') &&
    _hasMinigpuExport('_mgpuCreateSampler') &&
    _hasMinigpuExport('_mgpuWriteTexture2D') &&
    _hasMinigpuExport('_mgpuSetOwnedTexture') &&
    _hasMinigpuExport('_mgpuSetOwnedSampler') &&
    _hasMinigpuExport('_mgpuDestroyTexture') &&
    _hasMinigpuExport('_mgpuDestroyTextureView') &&
    _hasMinigpuExport('_mgpuDestroySampler') &&
    _mgpuTextureResourceVersion().toDartInt == 1;

MGPUTexture? mgpuCreateTexture2D(
  int width,
  int height,
  int format,
  int usage,
  int layers,
  int mipLevels,
) {
  if (!hasTextureResourceV1) return null;
  final texture = _mgpuCreateTexture2D(
    width.toJS,
    height.toJS,
    format.toJS,
    usage.toJS,
    layers.toJS,
    mipLevels.toJS,
  );
  return texture.toDartInt == 0 ? null : texture;
}

MGPUTextureView? mgpuCreateTextureView(
  MGPUTexture texture,
  int dimension,
  int baseMip,
  int mipCount,
  int baseLayer,
  int layerCount,
) {
  final view = _mgpuCreateTextureView(
    texture,
    dimension.toJS,
    baseMip.toJS,
    mipCount.toJS,
    baseLayer.toJS,
    layerCount.toJS,
  );
  return view.toDartInt == 0 ? null : view;
}

MGPUSampler? mgpuCreateSampler(
  int addressU,
  int addressV,
  int addressW,
  int mag,
  int min,
  int mip,
  double lodMin,
  double lodMax,
  int compare,
) {
  final sampler = _mgpuCreateSampler(
    addressU.toJS,
    addressV.toJS,
    addressW.toJS,
    mag.toJS,
    min.toJS,
    mip.toJS,
    lodMin.toJS,
    lodMax.toJS,
    compare.toJS,
  );
  return sampler.toDartInt == 0 ? null : sampler;
}

int mgpuWriteTexture2D(
  MGPUTexture texture,
  Uint8List bytes,
  int mipLevel,
  int arrayLayer,
) {
  final pointer = _malloc(bytes.lengthInBytes.toJS);
  final start = pointer.toDartInt;
  try {
    if (start <= 0 || start + bytes.lengthInBytes > _heapU8.length) {
      throw StateError('Texture upload allocation failed');
    }
    HEAPU8.set(Uint8ListToJSUint8Array(bytes).toJS, start);
    return _mgpuWriteTexture2D(
      texture,
      pointer,
      bytes.lengthInBytes.toJS,
      mipLevel.toJS,
      arrayLayer.toJS,
    ).toDartInt;
  } finally {
    _free(pointer);
  }
}

int mgpuWriteTextureRegion2D(
  MGPUTexture texture,
  Uint8List bytes,
  int x,
  int y,
  int width,
  int height,
  int mipLevel,
  int arrayLayer,
) {
  if (x < 0 ||
      x >= 32768 ||
      y < 0 ||
      y >= 32768 ||
      width <= 0 ||
      width > 32768 ||
      height <= 0 ||
      height > 32768 ||
      mipLevel < 0 ||
      mipLevel > 15 ||
      arrayLayer < 0 ||
      arrayLayer >= 32768 ||
      bytes.isEmpty ||
      bytes.lengthInBytes > 0x7fffffff) {
    throw ArgumentError('Invalid texture region upload');
  }
  if (!hasTextureRegionUploadV1) {
    throw UnsupportedError('Region uploads are unsupported by this module');
  }
  final pointer = _malloc(bytes.lengthInBytes.toJS);
  final start = pointer.toDartInt;
  try {
    if (start <= 0 || start + bytes.lengthInBytes > _heapU8.length) {
      throw StateError('Texture region upload allocation failed');
    }
    HEAPU8.set(Uint8ListToJSUint8Array(bytes).toJS, start);
    return _mgpuWriteTextureRegion2D(
      texture,
      pointer,
      bytes.lengthInBytes.toJS,
      x.toJS,
      y.toJS,
      width.toJS,
      height.toJS,
      mipLevel.toJS,
      arrayLayer.toJS,
    ).toDartInt;
  } finally {
    _free(pointer);
  }
}

int mgpuSetOwnedTexture(
  MGPUComputeShader shader,
  int slot,
  MGPUTextureView view,
  int kind,
) => _mgpuSetOwnedTexture(shader, slot.toJS, view, kind.toJS).toDartInt;

int mgpuSetOwnedSampler(
  MGPUComputeShader shader,
  int slot,
  MGPUSampler sampler,
  int kind,
) => _mgpuSetOwnedSampler(shader, slot.toJS, sampler, kind.toJS).toDartInt;

void mgpuDestroyTexture(MGPUTexture texture) => _mgpuDestroyTexture(texture);
void mgpuDestroyTextureView(MGPUTextureView view) =>
    _mgpuDestroyTextureView(view);
void mgpuDestroySampler(MGPUSampler sampler) => _mgpuDestroySampler(sampler);

@JS('_mgpuRasterResourceVersion')
external JSNumber _mgpuRasterResourceVersion();

@JS('_mgpuCreateRenderPipeline')
external MGPURenderPipeline _mgpuCreateRenderPipeline(
  JSNumber code,
  JSNumber vertex,
  JSNumber fragment,
  JSNumber words,
  JSNumber count,
);

@JS('_mgpuRenderPipelineStatus')
external JSNumber _mgpuRenderPipelineStatus(MGPURenderPipeline pipeline);

@JS('_mgpuRenderPipelineBindingStat')
external JSNumber _mgpuRenderPipelineBindingStat(
  MGPURenderPipeline pipeline,
  JSNumber selector,
);

@JS('_mgpuRenderPipelineDiagnostic')
external JSNumber _mgpuRenderPipelineDiagnostic(
  MGPURenderPipeline pipeline,
  JSNumber destination,
  JSNumber capacity,
);

@JS('_mgpuDestroyRenderPipeline')
external void _mgpuDestroyRenderPipeline(MGPURenderPipeline pipeline);

@JS('_mgpuSubmitRender')
external JSNumber _mgpuSubmitRender(
  MGPURenderPipeline pipeline,
  MGPUTextureView color,
  MGPUTextureView depth,
  JSNumber clear,
  JSNumber flags,
  JSNumber buffers,
  JSNumber bufferCount,
  JSNumber views,
  JSNumber viewCount,
  JSNumber samplers,
  JSNumber samplerCount,
  JSNumber commands,
  JSNumber commandCount,
);

const int _maxRasterDescriptorWords = 4096;
const int _maxRasterCommandWords = 65536;
const int _maxRasterResourcesPerFamily = 256;
const int _maxRasterDiagnosticBytes = 512;
const int _maxUint32 = 0xffffffff;

const List<String> _rasterExports = <String>[
  '_mgpuRasterResourceVersion',
  '_mgpuCreateRenderPipeline',
  '_mgpuRenderPipelineStatus',
  '_mgpuRenderPipelineDiagnostic',
  '_mgpuDestroyRenderPipeline',
  '_mgpuSubmitRender',
];

void _requireRasterResourceV1() {
  if (_rasterExports.any((name) => !_hasMinigpuExport(name)) ||
      _mgpuRasterResourceVersion().toDartInt != 1) {
    throw UnsupportedError('Raster resource ABI v1 is unavailable');
  }
}

bool get hasRasterResourceV1 {
  try {
    _requireRasterResourceV1();
    return true;
  } on UnsupportedError {
    return false;
  }
}

void _checkRasterWords(List<int> words, int maximum, String name) {
  if (words.length > maximum) {
    throw ArgumentError.value(words.length, name, 'Exceeds $maximum words');
  }
  for (final word in words) {
    if (word < 0 || word > _maxUint32) {
      throw ArgumentError.value(word, name, 'Must contain only uint32 values');
    }
  }
}

/// Checked [words] as one typed list, so they reach the heap in a single
/// TypedArray.set rather than one interop store per word.
Uint32List _rasterWords(List<int> words) =>
    words is Uint32List ? words : Uint32List.fromList(words);

JSNumber _checkedRasterMalloc(int byteLength) {
  if (byteLength <= 0) {
    throw ArgumentError.value(byteLength, 'byteLength', 'Must be positive');
  }
  final pointer = _malloc(byteLength.toJS);
  final rawStart = pointer.toDartDouble;
  if (!rawStart.isFinite ||
      rawStart <= 0 ||
      rawStart != rawStart.truncateToDouble() ||
      rawStart > _maxUint32) {
    throw StateError('Raster adapter allocation failed');
  }
  final start = rawStart.toInt();
  final heapLength = _heapU8.length;
  if (start <= 0 || start > heapLength || byteLength > heapLength - start) {
    if (start > 0) _free(pointer);
    throw StateError('Raster adapter allocation failed');
  }
  return pointer;
}

int _checkWasm32Handle(JSNumber handle, String name) {
  final value = handle.toDartDouble;
  if (!value.isFinite ||
      value <= 0 ||
      value != value.truncateToDouble() ||
      value > _maxUint32) {
    throw ArgumentError.value(value, name, 'Must be a wasm32 pointer');
  }
  return value.toInt();
}

JSNumber _renderUtf8(String value, List<JSNumber> allocations) {
  final bytes = utf8.encode(value);
  final pointer = _checkedRasterMalloc(bytes.length + 1);
  allocations.add(pointer);
  final start = pointer.toDartInt;
  HEAPU8.set(Uint8ListToJSUint8Array(bytes).toJS, start);
  _heapU8[start + bytes.length] = 0;
  return pointer;
}

MGPURenderPipeline? mgpuCreateRenderPipeline(
  String code,
  String vertex,
  String fragment,
  List<int> words,
) {
  _requireRasterResourceV1();
  final codeBytes = utf8.encode(code);
  final identifier = RegExp(r'^[A-Za-z_][A-Za-z0-9_]{0,63}$');
  if (codeBytes.isEmpty ||
      codeBytes.length > 1048576 ||
      code.contains('\u0000') ||
      !identifier.hasMatch(vertex) ||
      !identifier.hasMatch(fragment)) {
    throw ArgumentError('Invalid render pipeline strings');
  }
  _checkRasterWords(words, _maxRasterDescriptorWords, 'descriptor');
  final allocations = <JSNumber>[];
  try {
    final codePointer = _renderUtf8(code, allocations);
    final vertexPointer = _renderUtf8(vertex, allocations);
    final fragmentPointer = _renderUtf8(fragment, allocations);
    final wordPointer = _checkedRasterMalloc(words.length * 4);
    allocations.add(wordPointer);
    final start = wordPointer.toDartInt;
    if ((start & 3) != 0) {
      throw StateError('Raster descriptor allocation was not aligned');
    }
    final heap = _heapU32;
    final index = start ~/ 4;
    if (index > heap.length || words.length > heap.length - index) {
      throw StateError('Raster descriptor allocation exceeded the heap');
    }
    HEAPU32.set(Uint32ListToJSUint32Array(_rasterWords(words)).toJS, index);
    final result = _mgpuCreateRenderPipeline(
      codePointer,
      vertexPointer,
      fragmentPointer,
      wordPointer,
      words.length.toJS,
    );
    if (result.toDartDouble == 0) return null;
    _checkWasm32Handle(result, 'pipeline');
    return result;
  } finally {
    for (final pointer in allocations.reversed) {
      _free(pointer);
    }
  }
}

int mgpuRenderPipelineStatus(MGPURenderPipeline pipeline) {
  _requireRasterResourceV1();
  _checkWasm32Handle(pipeline, 'pipeline');
  return _mgpuRenderPipelineStatus(pipeline).toDartInt;
}

bool get hasRenderPipelineBindingStats =>
    _hasMinigpuExport('_mgpuRenderPipelineBindingStat');

int mgpuRenderPipelineBindingStat(
  MGPURenderPipeline pipeline,
  int selector,
) {
  _requireRasterResourceV1();
  if (!hasRenderPipelineBindingStats) {
    throw UnsupportedError('Render binding-cache diagnostics are unavailable');
  }
  _checkWasm32Handle(pipeline, 'pipeline');
  if (selector < 0 || selector > 3) {
    throw RangeError.range(selector, 0, 3, 'selector');
  }
  return _mgpuRenderPipelineBindingStat(
    pipeline,
    selector.toJS,
  ).toDartInt;
}

/// Whether this build calls `MinigpuModule.mgpuRenderPipelineSettled` when a
/// render pipeline settles. Older builds never call it, so callers poll.
bool get hasRenderPipelineSettledSignal =>
    _hasMinigpuExport('_mgpuRenderPipelineNotifySettledToPort');

/// Calls [onSettled] with a render pipeline's handle each time one settles
/// (ready, failed or closed). It may run inside a create or destroy call.
/// Replaces any earlier hook.
void setRenderPipelineSettledHook(void Function(int handle) onSettled) {
  final module = globalContext.getProperty<JSObject?>('MinigpuModule'.toJS);
  if (module == null || module.isUndefinedOrNull) {
    throw StateError('The minigpu wasm module is not loaded');
  }
  module.setProperty(
    'mgpuRenderPipelineSettled'.toJS,
    // Optional on purpose: dart2js dispatches a JS call on its argument
    // count, so a required parameter breaks a caller passing another number.
    (([JSNumber? handle]) {
      if (handle != null) onSettled(handle.toDartInt);
    }).toJS,
  );
}

String mgpuRenderPipelineDiagnostic(MGPURenderPipeline pipeline) {
  _requireRasterResourceV1();
  _checkWasm32Handle(pipeline, 'pipeline');
  final pointer = _checkedRasterMalloc(_maxRasterDiagnosticBytes + 1);
  try {
    final length = _mgpuRenderPipelineDiagnostic(
      pipeline,
      pointer,
      (_maxRasterDiagnosticBytes + 1).toJS,
    ).toDartInt;
    if (length < 0 || length > _maxRasterDiagnosticBytes) {
      return 'Render pipeline diagnostic was invalid';
    }
    final start = pointer.toDartInt;
    final heap = _heapU8;
    if (start > heap.length || length > heap.length - start) {
      return 'Render pipeline diagnostic was invalid';
    }
    return utf8.decode(
      heap.sublist(start, start + length),
      allowMalformed: true,
    );
  } finally {
    _free(pointer);
  }
}

void mgpuDestroyRenderPipeline(MGPURenderPipeline pipeline) {
  _requireRasterResourceV1();
  _checkWasm32Handle(pipeline, 'pipeline');
  _mgpuDestroyRenderPipeline(pipeline);
}

int mgpuSubmitRender(
  MGPURenderPipeline pipeline,
  MGPUTextureView color,
  MGPUTextureView? depth,
  List<double> clear,
  int flags,
  List<MGPUBuffer> buffers,
  List<MGPUTextureView> views,
  List<MGPUSampler> samplers,
  List<int> commands,
) {
  _requireRasterResourceV1();
  _checkWasm32Handle(pipeline, 'pipeline');
  if (clear.length != 5) {
    throw ArgumentError.value(clear, 'clear', 'Must contain RGBA and depth');
  }
  final clear32 = Float32List(5);
  for (var i = 0; i < clear.length; i++) {
    final value = clear[i];
    if (!value.isFinite) {
      throw ArgumentError.value(value, 'clear', 'Must be finite');
    }
    clear32[i] = value;
    if (!clear32[i].isFinite) {
      throw ArgumentError.value(
        value,
        'clear',
        'Must be representable as float32',
      );
    }
  }
  if (clear[4] < 0 || clear[4] > 1) {
    throw ArgumentError.value(clear[4], 'clear', 'Depth must be in 0..1');
  }
  if (flags < 0 || flags > 3 || (depth == null && (flags & 2) != 0)) {
    throw ArgumentError.value(flags, 'flags', 'Invalid render load flags');
  }
  if (buffers.length > _maxRasterResourcesPerFamily ||
      views.length > _maxRasterResourcesPerFamily ||
      samplers.length > _maxRasterResourcesPerFamily) {
    throw ArgumentError('Render resource count exceeds 256');
  }
  _checkRasterWords(commands, _maxRasterCommandWords, 'commands');
  _checkWasm32Handle(color, 'color');
  for (final handle in buffers) {
    _checkWasm32Handle(handle, 'buffers');
  }
  for (final handle in views) {
    _checkWasm32Handle(handle, 'views');
  }
  for (final handle in samplers) {
    _checkWasm32Handle(handle, 'samplers');
  }
  if (depth != null) {
    _checkWasm32Handle(depth, 'depth');
  }

  final allocations = <JSNumber>[];
  JSNumber allocate(int byteLength) {
    final pointer = _checkedRasterMalloc(byteLength);
    allocations.add(pointer);
    if ((pointer.toDartInt & 3) != 0) {
      throw StateError('Raster allocation was not 4-byte aligned');
    }
    return pointer;
  }

  try {
    final clearPointer = allocate(5 * 4);
    final bufferPointer = buffers.isEmpty
        ? 0.toJS
        : allocate(buffers.length * 4);
    final viewPointer = views.isEmpty ? 0.toJS : allocate(views.length * 4);
    final samplerPointer = samplers.isEmpty
        ? 0.toJS
        : allocate(samplers.length * 4);
    final commandPointer = commands.isEmpty
        ? 0.toJS
        : allocate(commands.length * 4);

    final heapBytes = _heapU8;
    final clearStart = clearPointer.toDartInt;
    if (clearStart > heapBytes.length || 20 > heapBytes.length - clearStart) {
      throw StateError('Raster clear allocation exceeded the heap');
    }
    HEAPF32.set(Float32ListToJSFloat32Array(clear32).toJS, clearStart ~/ 4);

    final heapWords = _heapU32;
    void writeHandles(JSNumber pointer, List<JSNumber> handles) {
      if (handles.isEmpty) return;
      final index = pointer.toDartInt ~/ 4;
      if (index > heapWords.length ||
          handles.length > heapWords.length - index) {
        throw StateError('Raster pointer array exceeded the heap');
      }
      for (var i = 0; i < handles.length; i++) {
        heapWords[index + i] = _checkWasm32Handle(handles[i], 'handles');
      }
    }

    writeHandles(bufferPointer, buffers);
    writeHandles(viewPointer, views);
    writeHandles(samplerPointer, samplers);
    if (commands.isNotEmpty) {
      final index = commandPointer.toDartInt ~/ 4;
      if (index > heapWords.length ||
          commands.length > heapWords.length - index) {
        throw StateError('Raster command array exceeded the heap');
      }
      HEAPU32.set(
        Uint32ListToJSUint32Array(_rasterWords(commands)).toJS,
        index,
      );
    }
    return _mgpuSubmitRender(
      pipeline,
      color,
      depth ?? 0.toJS,
      clearPointer,
      flags.toJS,
      bufferPointer,
      buffers.length.toJS,
      viewPointer,
      views.length.toJS,
      samplerPointer,
      samplers.length.toJS,
      commandPointer,
      commands.length.toJS,
    ).toDartInt;
  } finally {
    for (final pointer in allocations.reversed) {
      _free(pointer);
    }
  }
}

@JS('_mgpuDestroyBuffer')
external void _mgpuDestroyBuffer(MGPUBuffer buffer);

void mgpuDestroyBuffer(MGPUBuffer buffer) {
  // The browser build frees the Buffer inline, so a read asleep on it (or
  // queued behind one) would resume into freed memory: while any suspending
  // call is admitted, the free takes its turn after them.
  if (_suspendingCalls == 0) {
    _mgpuDestroyBuffer(buffer);
  } else {
    _serialAsync<void>(() async => _mgpuDestroyBuffer(buffer)).ignore();
  }
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
  // A direct export call, not a queued ccall: dispatch only records into the
  // open batch and never suspends, and it must stay in program order with the
  // synchronous setBuffer calls around it.
  _mgpuDispatch(shader, groupsX.toJS, groupsY.toJS, groupsZ.toJS);
}

/// Whether the loaded module can compile a pipeline without dispatching.
bool get hasComputePipelineBuild =>
    _hasMinigpuExport('_mgpuComputeShaderBuildPipeline');

@JS('_mgpuComputeShaderBuildPipeline')
external JSNumber _mgpuComputeShaderBuildPipeline(MGPUComputeShader shader);

/// Compiles [shader]'s pipeline for its kernel and bindings without
/// dispatching; true when it exists. A direct export call like [mgpuDispatch]:
/// the compile is the one a first dispatch does, and it never suspends.
bool mgpuComputeShaderBuildPipeline(MGPUComputeShader shader) =>
    _mgpuComputeShaderBuildPipeline(shader).toDartInt == 1;

@JS('ccall')
external JSPromise ccall(
  JSString name,
  JSString returnType,
  JSArray argTypes,
  JSArray args,
  JSObject opts,
);

extension _HeapCopy on JSTypedArray {
  // TypedArray.prototype.set: one bulk copy into a heap view on both
  // compilers. A JS-backed heap list's setRange from a dart2wasm list goes
  // element by element.
  external void set(JSTypedArray source, int targetOffset);
}

@JS('_mgpuDispatch')
external void _mgpuDispatch(
  MGPUComputeShader shader,
  JSNumber groupsX,
  JSNumber groupsY,
  JSNumber groupsZ,
);

@JS('_mgpuWriteFloat')
external void _mgpuWriteFloat(
  MGPUBuffer buffer,
  JSNumber inputDataPtr,
  JSNumber size,
);

/// The JS GPUBuffer's byte size, or null when its object cannot be resolved.
int? _gpuBufferBytes(MGPUBuffer buffer) {
  final target = getWebGpuJsObject(mgpuGetWGPUBufferHandle(buffer));
  return target == null ? null : (target as _SizedGPUBuffer).size;
}

// MGPUElementType codes (minigpu.h): the element type mgpuReadSyncChecked reads.
const int _elemI8 = 0, _elemU8 = 1, _elemI16 = 2, _elemU16 = 3, _elemI32 = 4,
    _elemU32 = 5, _elemI64 = 6, _elemU64 = 7, _elemF32 = 8, _elemF64 = 9;

/// Reads [readElements] (0: the whole list) [elementSize]-byte elements of
/// [elementType] from GPU element [elementOffset] into the FRONT of
/// [outputData], as the native backend does; a range that does not fit throws
/// rather than reading short, and a read that fails in C (no device, a failed
/// map) throws a [StateError] rather than handing back stale heap bytes.
/// [floatWidth] is the float width, 0 for an integer type.
Future<void> _readAsync(
  int elementType,
  MGPUBuffer buffer,
  TypedData outputData,
  int elementSize,
  int readElements,
  int elementOffset, {
  int floatWidth = 0,
}) async {
  final int capacity = outputData.lengthInBytes ~/ elementSize;
  final int count = readElements > 0 ? readElements : capacity;
  if (elementOffset < 0) {
    throw RangeError.value(elementOffset, 'elementOffset', 'Cannot be negative');
  }
  if (count > capacity) {
    throw RangeError(
      'Read range (offset: $elementOffset, count: $count) exceeds output '
      'buffer capacity ($capacity elements)',
    );
  }
  if (count == 0) return;
  final int byteLength = count * elementSize;
  // C refuses an out-of-range read too; checked here it is a RangeError.
  final int? gpuBytes = _gpuBufferBytes(buffer);
  if (gpuBytes != null &&
      (elementOffset > gpuBytes ||
          (elementOffset + count) * elementSize > gpuBytes)) {
    throw RangeError(
      'Read range (offset: $elementOffset, count: $count) exceeds the GPU '
      'buffer ($gpuBytes bytes)',
    );
  }
  await _serialAsync<void>(() async {
    final JSNumber ptr = _malloc(byteLength.toJS);
    final int start = ptr.toDartInt;
    try {
      if (start <= 0 || start + byteLength > _heapU8.length) {
        throw StateError('No wasm heap scratch for a $byteLength-byte read');
      }
      final filled = await ccall(
        'mgpuReadSyncChecked'.toJS,
        "number".toJS,
        ["number", "number", "number", "number", "number"].toJS,
        [buffer, ptr, count.toJS, elementOffset.toJS, elementType.toJS].toJS,
        {"async": true}.toJSDeep,
      ).toDart;
      if ((filled as JSNumber?)?.toDartInt != 1) {
        throw StateError(
          'GPU buffer read failed (offset: $elementOffset, count: $count)',
        );
      }
      // The heap views are read again here: the read can grow wasm memory.
      _copyFromHeap(start, outputData, byteLength, floatWidth);
    } finally {
      _free(ptr);
    }
  });
}

/// Copies [byteLength] heap bytes at [start] to the front of [out], in one
/// bulk copy on both compilers when a heap view of [out]'s own type fits.
/// Integer views are bit-exact; a float view moves numbers, so it is used for
/// float data only. Anything else (64-bit, ByteData) goes as bytes.
void _copyFromHeap(int start, TypedData out, int byteLength, int floatWidth) {
  final int width = out.elementSizeInBytes;
  final bool whole = start % width == 0 && byteLength % width == 0;
  final int from = start ~/ width, n = byteLength ~/ width;
  if (whole && out is Uint8List) {
    out.setRange(0, n, _heapU8, from);
  } else if (whole && out is Int8List) {
    out.setRange(0, n, _heapI8, from);
  } else if (whole && out is Uint16List) {
    out.setRange(0, n, _heapU16, from);
  } else if (whole && out is Int16List) {
    out.setRange(0, n, _heapI16, from);
  } else if (whole && out is Uint32List) {
    out.setRange(0, n, _heapU32, from);
  } else if (whole && out is Int32List) {
    out.setRange(0, n, _heapI32, from);
  } else if (whole && floatWidth == 4 && out is Float32List) {
    out.setRange(0, n, _heapF32, from);
  } else if (whole && floatWidth == 8 && out is Float64List) {
    out.setRange(0, n, _heapF64, from);
  } else {
    out.buffer
        .asUint8List(out.offsetInBytes, byteLength)
        .setRange(0, byteLength, _heapU8, start);
  }
}

/// The write-side twin of [_copyFromHeap]: [byteLength] bytes from the front
/// of [data] to heap byte [start], with the same choice of views.
void _copyToHeap(TypedData data, int start, int byteLength, int floatWidth) {
  final int width = data.elementSizeInBytes;
  final bool whole = start % width == 0 && byteLength % width == 0;
  final int to = start ~/ width, n = byteLength ~/ width;
  if (whole && data is Uint8List) {
    HEAPU8.set(
        Uint8ListToJSUint8Array(Uint8List.sublistView(data, 0, n)).toJS, to);
  } else if (whole && data is Int8List) {
    HEAP8.set(Int8ListToJSInt8Array(Int8List.sublistView(data, 0, n)).toJS, to);
  } else if (whole && data is Uint16List) {
    HEAPU16.set(
        Uint16ListToJSUint16Array(Uint16List.sublistView(data, 0, n)).toJS,
        to);
  } else if (whole && data is Int16List) {
    HEAP16.set(
        Int16ListToJSInt16Array(Int16List.sublistView(data, 0, n)).toJS, to);
  } else if (whole && data is Uint32List) {
    HEAPU32.set(
        Uint32ListToJSUint32Array(Uint32List.sublistView(data, 0, n)).toJS,
        to);
  } else if (whole && data is Int32List) {
    HEAP32.set(
        Int32ListToJSInt32Array(Int32List.sublistView(data, 0, n)).toJS, to);
  } else if (whole && floatWidth == 4 && data is Float32List) {
    HEAPF32.set(
        Float32ListToJSFloat32Array(Float32List.sublistView(data, 0, n)).toJS,
        to);
  } else if (whole && floatWidth == 8 && data is Float64List) {
    HEAPF64.set(
        Float64ListToJSFloat64Array(Float64List.sublistView(data, 0, n)).toJS,
        to);
  } else {
    final bytes = data.buffer.asUint8List(data.offsetInBytes, byteLength);
    HEAPU8.set(Uint8ListToJSUint8Array(bytes).toJS, start);
  }
}

/// Uploads [size] (0: the whole list) [elementSize]-byte elements from the
/// front of [inputData] through [upload], a C write export taking a heap
/// pointer and a byte count. Any list type uploads its bytes, as natively; a
/// count the list cannot supply throws instead of uploading stale heap bytes.
void _writeSync(
  void Function(MGPUBuffer buffer, JSNumber dataPtr, JSNumber byteSize) upload,
  MGPUBuffer buffer,
  TypedData inputData,
  int elementSize,
  int size, {
  int floatWidth = 0,
}) {
  final int capacity = inputData.lengthInBytes ~/ elementSize;
  final int count = size > 0 ? size : capacity;
  if (count > capacity) {
    throw RangeError(
      'elementCount ($count) exceeds input data capacity ($capacity elements)',
    );
  }
  if (count == 0) return;
  final int byteLength = count * elementSize;
  // C throws past the buffer's end (packed types pad to a word), and this
  // wasm build cannot catch a C++ exception.
  final int? gpuBytes = _gpuBufferBytes(buffer);
  if (gpuBytes != null && (byteLength + 3) ~/ 4 * 4 > gpuBytes) {
    throw RangeError(
      'Write of $byteLength bytes exceeds the GPU buffer ($gpuBytes bytes)',
    );
  }
  final JSNumber ptr = _malloc(byteLength.toJS);
  final int start = ptr.toDartInt;
  try {
    if (start <= 0 || start + byteLength > _heapU8.length) {
      throw StateError('No wasm heap scratch for a $byteLength-byte write');
    }
    _copyToHeap(inputData, start, byteLength, floatWidth);
    upload(buffer, ptr, byteLength.toJS);
  } finally {
    _free(ptr);
  }
}

/// Asynchronous reads, one per element type: [readElements] (0: the whole
/// list) elements from GPU element [elementOffset] land at the front of
/// [outputData], whatever its list type. See [_readAsync].

Future<void> mgpuReadAsyncInt8(
  MGPUBuffer buffer,
  TypedData outputData, {
  int readElements = 0,
  int elementOffset = 0,
}) => _readAsync(_elemI8, buffer, outputData, 1, readElements,
    elementOffset);

Future<void> mgpuReadAsyncInt16(
  MGPUBuffer buffer,
  TypedData outputData, {
  int readElements = 0,
  int elementOffset = 0,
}) => _readAsync(_elemI16, buffer, outputData, 2, readElements,
    elementOffset);

Future<void> mgpuReadAsyncInt32(
  MGPUBuffer buffer,
  TypedData outputData, {
  int readElements = 0,
  int elementOffset = 0,
}) => _readAsync(_elemI32, buffer, outputData, 4, readElements,
    elementOffset);

// Any list or ByteData: the 8-byte elements are copied as bytes, so an
// Int64List (dart2wasm) or a byte view (dart2js has none) gets them exactly.
Future<void> mgpuReadAsyncInt64(
  MGPUBuffer buffer,
  TypedData outputData, {
  int readElements = 0,
  int elementOffset = 0,
}) => _readAsync(_elemI64, buffer, outputData, 8, readElements,
    elementOffset);

Future<void> mgpuReadAsyncUint8(
  MGPUBuffer buffer,
  TypedData outputData, {
  int readElements = 0,
  int elementOffset = 0,
}) => _readAsync(_elemU8, buffer, outputData, 1, readElements,
    elementOffset);

Future<void> mgpuReadAsyncUint16(
  MGPUBuffer buffer,
  TypedData outputData, {
  int readElements = 0,
  int elementOffset = 0,
}) => _readAsync(_elemU16, buffer, outputData, 2, readElements,
    elementOffset);

Future<void> mgpuReadAsyncUint32(
  MGPUBuffer buffer,
  TypedData outputData, {
  int readElements = 0,
  int elementOffset = 0,
}) => _readAsync(_elemU32, buffer, outputData, 4, readElements,
    elementOffset);

Future<void> mgpuReadAsyncUint64(
  MGPUBuffer buffer,
  TypedData outputData, {
  int readElements = 0,
  int elementOffset = 0,
}) => _readAsync(_elemU64, buffer, outputData, 8, readElements,
    elementOffset);

Future<void> mgpuReadAsyncFloat(
  MGPUBuffer buffer,
  TypedData outputData, {
  int readElements = 0,
  int elementOffset = 0,
}) => _readAsync(_elemF32, buffer, outputData, 4, readElements,
    elementOffset, floatWidth: 4);

Future<void> mgpuReadAsyncDouble(
  MGPUBuffer buffer,
  TypedData outputData, {
  int readElements = 0,
  int elementOffset = 0,
}) => _readAsync(_elemF64, buffer, outputData, 8, readElements,
    elementOffset, floatWidth: 8);

@JS('_mgpuWriteInt8')
external void _mgpuWriteInt8(
  MGPUBuffer buffer,
  JSNumber inputDataPtr,
  JSNumber size,
);

/// Writes [size] (0: the whole list) elements from the front of [inputData],
/// whatever its list type; see [_writeSync].
void mgpuWriteInt8(MGPUBuffer buffer, TypedData inputData, int size) =>
    _writeSync((b, p, n) => _mgpuWriteInt8(b, p, n), buffer, inputData, 1,
        size);

@JS('_mgpuWriteInt16')
external void _mgpuWriteInt16(
  MGPUBuffer buffer,
  JSNumber inputDataPtr,
  JSNumber size,
);

void mgpuWriteInt16(MGPUBuffer buffer, TypedData inputData, int size) =>
    _writeSync((b, p, n) => _mgpuWriteInt16(b, p, n), buffer, inputData, 2,
        size);

@JS('_mgpuWriteInt32')
external void _mgpuWriteInt32(
  MGPUBuffer buffer,
  JSNumber inputDataPtr,
  JSNumber size,
);

void mgpuWriteInt32(MGPUBuffer buffer, TypedData inputData, int size) =>
    _writeSync((b, p, n) => _mgpuWriteInt32(b, p, n), buffer, inputData, 4,
        size);

@JS('_mgpuWriteInt64')
external void _mgpuWriteInt64(
  MGPUBuffer buffer,
  JSNumber inputDataPtr,
  JSNumber size,
);

// 64-bit elements go as bytes: dart2js has no Int64List, and a ByteData or
// byte view of them uploads exactly.
void mgpuWriteInt64(MGPUBuffer buffer, TypedData inputData, int size) =>
    _writeSync((b, p, n) => _mgpuWriteInt64(b, p, n), buffer, inputData, 8,
        size);

@JS('_mgpuWriteUint8')
external void _mgpuWriteUint8(
  MGPUBuffer buffer,
  JSNumber inputDataPtr,
  JSNumber size,
);

void mgpuWriteUint8(MGPUBuffer buffer, TypedData inputData, int size) =>
    _writeSync((b, p, n) => _mgpuWriteUint8(b, p, n), buffer, inputData, 1,
        size);

@JS('_mgpuWriteUint16')
external void _mgpuWriteUint16(
  MGPUBuffer buffer,
  JSNumber inputDataPtr,
  JSNumber size,
);

void mgpuWriteUint16(MGPUBuffer buffer, TypedData inputData, int size) =>
    _writeSync((b, p, n) => _mgpuWriteUint16(b, p, n), buffer, inputData, 2,
        size);

@JS('_mgpuWriteUint32')
external void _mgpuWriteUint32(
  MGPUBuffer buffer,
  JSNumber inputDataPtr,
  JSNumber size,
);

void mgpuWriteUint32(MGPUBuffer buffer, TypedData inputData, int size) =>
    _writeSync((b, p, n) => _mgpuWriteUint32(b, p, n), buffer, inputData, 4,
        size);

@JS('_mgpuWriteUint64')
external void _mgpuWriteUint64(
  MGPUBuffer buffer,
  JSNumber inputDataPtr,
  JSNumber size,
);

void mgpuWriteUint64(MGPUBuffer buffer, TypedData inputData, int size) =>
    _writeSync((b, p, n) => _mgpuWriteUint64(b, p, n), buffer, inputData, 8,
        size);

Future<void> mgpuWriteFloat(
  MGPUBuffer buffer,
  TypedData inputData,
  int size,
) async {
  // A direct export call like the other writes: C's write never suspends, and
  // a queued ccall could fall behind a sleeping read and land after later
  // writes and dispatches.
  _writeSync((b, p, n) => _mgpuWriteFloat(b, p, n), buffer, inputData, 4,
      size, floatWidth: 4);
}

@JS('_mgpuWriteDouble')
external void _mgpuWriteDouble(
  MGPUBuffer buffer,
  JSNumber inputDataPtr,
  JSNumber size,
);

void mgpuWriteDouble(MGPUBuffer buffer, TypedData inputData, int size) =>
    _writeSync((b, p, n) => _mgpuWriteDouble(b, p, n), buffer, inputData, 8,
        size, floatWidth: 8);

// ---------------------------------------------------------------------------
// Web video texture � GPUDevice.importExternalTexture
// ---------------------------------------------------------------------------
// The VideoFrame JS object is passed as a JSAny handle.
// GPUExternalTexture is valid only for the current task (microtask boundary).

@JS()
@staticInterop
class GPUDevice {}

extension GPUDeviceImport on GPUDevice {
  external JSObject importExternalTexture(JSObject descriptor);

  // The rest of the device surface needed to run a compute pass from Dart.
  // Declared here rather than pulled from a WebGPU package so this file keeps
  // its single dependency on `dart:js_interop`.
  external JSObject createShaderModule(JSObject descriptor);
  external JSObject createComputePipeline(JSObject descriptor);
  external JSObject createBindGroup(JSObject descriptor);
  external JSObject createCommandEncoder();
  external JSObject get queue;
}

@JS()
@staticInterop
class _GPUPipeline {}

extension _GPUPipelineExt on _GPUPipeline {
  external JSObject getBindGroupLayout(int index);
}

@JS()
@staticInterop
class _GPUQueue {}

extension _GPUQueueExt on _GPUQueue {
  external void submit(JSArray<JSObject> buffers);
  external JSPromise<JSAny?> onSubmittedWorkDone();
  external void writeBuffer(
    JSObject buffer,
    int bufferOffset,
    JSUint8Array data,
  );
}

@JS()
@staticInterop
class _SizedGPUBuffer {}

extension _SizedGPUBufferExt on _SizedGPUBuffer {
  external int get size;
}

/// Upload exactly the caller's byte view into minigpu's own device/buffer.
/// The typed-array overload preserves a subview's offset and length. Flushing
/// the open compute batch orders earlier work before this queue write.
void mgpuWriteRawBytesAt(MGPUBuffer buffer, Uint8List bytes, int offset) {
  if (offset < 0 || offset % 4 != 0 || bytes.lengthInBytes % 4 != 0) {
    throw ArgumentError('Raw upload offset/length must be aligned to 4 bytes');
  }
  final target = getWebGpuJsObject(mgpuGetWGPUBufferHandle(buffer));
  final device = getWebGpuJsObject(mgpuGetWGPUDeviceHandle());
  if (target == null || device == null)
    throw StateError('GPU buffer/device unavailable');
  if (offset + bytes.lengthInBytes > (target as _SizedGPUBuffer).size) {
    throw RangeError('Raw upload exceeds GPU buffer');
  }
  if (bytes.isEmpty) return;
  mgpuFlushBatch();
  final typedBytes = Uint8ListToJSUint8Array(bytes).toJS;
  ((device as GPUDevice).queue as _GPUQueue).writeBuffer(
    target,
    offset,
    typedBytes,
  );
}

@JS()
@staticInterop
class _GPUEncoder {}

extension _GPUEncoderExt on _GPUEncoder {
  external JSObject beginComputePass();
  external JSObject finish();
}

@JS()
@staticInterop
class _GPUComputePass {}

extension _GPUComputePassExt on _GPUComputePass {
  external void setPipeline(JSObject pipeline);
  external void setBindGroup(int index, JSObject bindGroup);
  external void dispatchWorkgroups(int x, int y, int z);
  external void end();
}

/// The landing kernel, in the ONE form a `GPUExternalTexture` can be read.
///
/// 🔴 `texture_external`, NOT `texture_2d<f32>` — binding an external texture
/// to a 2D-texture declaration fails pipeline validation with
/// "Binding doesn't exist in [BindGroupLayoutInternal]", and because WebGPU
/// validation errors are UNCAPTURED and ASYNCHRONOUS the dispatch then fails
/// silently on every frame. That is a black screen under a healthy log.
///
/// `textureLoad` accepts a `texture_external` directly (two args, no mip
/// level), so no sampler and no bind-group entry for one. The browser has
/// already done any YUV->RGB conversion, so `.x` is red exactly as in the
/// native landing kernel — which is what keeps this byte-identical to it.
///
/// 🔴 GEOMETRY IS BAKED IN AS `const`, NOT PASSED IN A UNIFORM. The uniform
/// version needed `GPUQueue.writeBuffer`, which failed on Chrome with
/// "Overload resolution failed" through this interop layer — so the whole
/// call, and the buffer it needed, are gone. Resolution changes are rare
/// (once per capture configuration), a shader compile is cheap, and pipelines
/// are cached per geometry. Fewer moving interop pieces on a path that cannot
/// be unit-tested is worth more than avoiding a recompile nobody notices.
String _webLandWgsl(int w, int h, int s) =>
    '''
@group(0) @binding(0) var srcTex: texture_external;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

const W: u32 = ${w}u;
const H: u32 = ${h}u;
const S: u32 = ${s}u;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= W || gid.y >= H) { return; }
  var c: vec4<f32>;
  if (S <= 1u) {
    c = textureLoad(srcTex, vec2<i32>(i32(gid.x), i32(gid.y)));
  } else {
    var acc = vec4<f32>(0.0);
    for (var yy = 0u; yy < S; yy = yy + 1u) {
      for (var xx = 0u; xx < S; xx = xx + 1u) {
        acc = acc + textureLoad(srcTex,
            vec2<i32>(i32(gid.x * S + xx), i32(gid.y * S + yy)));
      }
    }
    c = acc / f32(S * S);
  }
  let q = clamp(c, vec4<f32>(0.0), vec4<f32>(1.0)) * 255.0 + vec4<f32>(0.5);
  dst[gid.y * W + gid.x] =
      u32(q.x) | (u32(q.y) << 8u) | (u32(q.z) << 16u) | (u32(q.w) << 24u);
}
''';

/// Landing pipelines, keyed by geometry, all made on [_landDevice]: a pipeline
/// is valid only on the device that created it, and a replaced context brings
/// a new device. Small and bounded: one entry per distinct capture
/// configuration seen on that device.
final Map<String, JSObject> _landPipelines = {};
JSObject? _landDevice;

/// Run the external-texture landing pass: [externalTexture] -> packed RGBA8
/// `array<u32>` in the buffer behind [dstBufferHandle], the same destination
/// layout the native D3D11 path writes.
///
/// Returns false — having said why, once — if any step fails. 🔴 A false here
/// must NOT be treated as "try the ordinary route instead": no other route on
/// this platform can read an external texture.
bool mgpuLandExternalTexture({
  required JSObject externalTexture,
  required int dstBufferHandle,
  required int width,
  required int height,
  required int downscale,
}) {
  final dev = _importDevice();
  if (dev == null) {
    _sayLandFailed('no GPUDevice');
    return false;
  }
  final dst = getWebGpuJsObject(dstBufferHandle);
  if (dst == null) {
    _sayLandFailed(
      'destination WGPUBuffer $dstBufferHandle is not in the '
      'Emscripten object table',
    );
    return false;
  }
  final device = dev as JSObject;
  if (!device.strictEquals(_landDevice).toDart) {
    _landPipelines.clear();
    _landDevice = device;
  }

  final s = downscale < 1 ? 1 : downscale;
  final key = '${width}x${height}:$s';
  var step = 'createComputePipeline';
  try {
    final pipeline = _landPipelines.putIfAbsent(key, () {
      final module = dev.createShaderModule(
        {'code': _webLandWgsl(width, height, s)}.jsify() as JSObject,
      );
      return dev.createComputePipeline(
        {
              'layout': 'auto',
              'compute': {'module': module, 'entryPoint': 'main'},
            }.jsify()
            as JSObject,
      );
    });

    // 🔴 THE BIND GROUP CANNOT BE CACHED. A GPUExternalTexture is SINGLE-USE:
    // it expires with the frame it came from, so a cached group would
    // reference a dead texture on the very next pass.
    step = 'createBindGroup';
    final bindGroup = dev.createBindGroup(
      {
            'layout': (pipeline as _GPUPipeline).getBindGroupLayout(0),
            'entries': [
              {'binding': 0, 'resource': externalTexture},
              {
                'binding': 1,
                'resource': {'buffer': dst},
              },
            ],
          }.jsify()
          as JSObject,
    );

    step = 'dispatch';
    final encoder = dev.createCommandEncoder() as _GPUEncoder;
    final pass = encoder.beginComputePass() as _GPUComputePass;
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups((width + 7) ~/ 8, (height + 7) ~/ 8, 1);
    pass.end();
    // Dispatches minigpu recorded earlier sit in its open batch until a flush;
    // they must reach the queue before this pass rewrites dst, not after.
    mgpuFlushBatch();
    (dev.queue as _GPUQueue).submit(<JSObject>[encoder.finish()].toJS);
    return true;
  } catch (e) {
    // The step is named because these are all one-line interop calls that fail
    // identically from the outside, and this path cannot be unit-tested.
    _sayLandFailed('$step threw: $e');
    _landPipelines.remove(key);
    return false;
  }
}

bool _saidLandFailed = false;

void _sayLandFailed(String why) {
  if (_saidLandFailed) return;
  _saidLandFailed = true;
  // ignore: avoid_print
  print(
    '[minigpu] 🔴 external-texture landing pass FAILED — $why. Zero-copy '
    'video capture will produce nothing on this page.',
  );
}

// The JS global 'gpuDevice' must be set by the app before calling
// mgpuImportVideoFrameWeb(). In a Flutter Web app this is the
// device retrieved from gpu_pipeline or a direct WebGPU init.
@JS('gpuDevice')
external GPUDevice? get jsGpuDevice;

/// Import a `VideoFrame` (or `HTMLVideoElement`) as a `GPUExternalTexture`.
///
/// 🔴 **THE DEVICE IS OURS, NOT THE HOST PAGE'S.** This used to read ONLY the
/// JS global `globalThis.gpuDevice` and return null when it was unset — and nothing
/// in this repo sets it, so on a page where the wasm side never ran its
/// `EM_ASM` setter every import failed silently. Observed live as a camera that
/// delivered 121 VideoFrames of which NONE landed, with no error anywhere: a
/// null here is indistinguishable from "the browser rejected the frame".
///
/// minigpu already knows its own device — `mgpuGetWGPUDeviceHandle()` is the
/// Emscripten handle and `getWebGpuJsObject` resolves it — so ask for it
/// directly and treat the global as a legacy override only. It must be the SAME
/// device that will sample the texture, which is exactly what this returns.
JSObject? mgpuImportExternalTexture(JSAny videoFrame) {
  final dev = _importDevice();
  if (dev == null) {
    _sayImportFailed(
      'no GPUDevice — minigpu is not initialised on this page '
      '(mgpuGetWGPUDeviceHandle returned 0 and globalThis.gpuDevice is '
      'unset)',
    );
    return null;
  }
  final desc = {'source': videoFrame}.toJSDeep as JSObject;
  try {
    return dev.importExternalTexture(desc);
  } catch (e) {
    // Real rejections land here: a closed VideoFrame, an unsupported colour
    // space, a source this browser will not wrap. Distinguishable from the
    // missing-device case above, which is the whole point of splitting them.
    _sayImportFailed('importExternalTexture threw: $e');
    return null;
  }
}

/// Our own device first; the host-page global only as a legacy override.
GPUDevice? _importDevice() {
  try {
    final own = getWebGpuJsObject(mgpuGetWGPUDeviceHandle());
    if (own != null) return own as GPUDevice;
  } catch (_) {
    // Module not loaded yet, or no WebGPU global — fall through.
  }
  return jsGpuDevice;
}

bool _saidImportFailed = false;

/// Named ONCE. A per-frame message here would be thousands of lines a second,
/// and silence is what made this cost a debugging round in the first place.
void _sayImportFailed(String why) {
  if (_saidImportFailed) return;
  _saidImportFailed = true;
  // ignore: avoid_print
  print(
    '[minigpu] 🔴 importExternalTexture FAILED — $why. Every zero-copy '
    'video frame will fall back to the CPU path until this is fixed.',
  );
}

// ---------------------------------------------------------------------------
// Emscripten WASM exports — device and buffer handles
// ---------------------------------------------------------------------------
// Note: getWebGpuJsObject() lives in webgpu_interop.dart (no @JS('Module')
// scope) so it accesses the global window.WebGPU object, not Module.WebGPU.

@JS('_mgpuSetLogLevel')
external void _mgpuSetLogLevelJs(JSNumber level);

/// Minimum native log verbosity (0=DEBUG 1=INFO 2=WARN 3=ERROR, -1 silences
/// all). The wasm default is INFO, which narrates ~7 buffer ops per encoded
/// frame straight onto the browser console via stderr `printChar`. Callable
/// only once the module is loaded — `MinigpuWeb` stages the level and applies
/// it right after [ensureMinigpuModuleLoaded], before context init, so even
/// the adapter/device bring-up logs honor it.
void mgpuSetLogLevel(int level) => _mgpuSetLogLevelJs(level.toJS);

@JS('_mgpuGetWGPUDeviceHandle')
external JSNumber _mgpuGetWGPUDeviceHandleJs();

/// Returns the Emscripten integer handle for the WGPUDevice.
/// Pass to [getWebGpuJsObject] from webgpu_interop.dart to get the JS GPUDevice.
int mgpuGetWGPUDeviceHandle() =>
    _mgpuGetWGPUDeviceHandleJs().toDartDouble.toInt();

/// Resolves minigpu's own device. Never uses the host-page gpuDevice global.
JSObject? mgpuGetOwnDeviceObject() {
  if (!isMinigpuModuleLoaded ||
      !_hasMinigpuExport('_mgpuGetWGPUDeviceHandle')) return null;
  try {
    return getWebGpuJsObject(mgpuGetWGPUDeviceHandle());
  } catch (_) {
    return null;
  }
}

/// Receipt from the exact loaded module's creation context. Captured functions
/// cannot accidentally poll a later module whose token numbers happen to match.
final class WebContextRetirement {
  WebContextRetirement._(this._module, this._status, this._release, this._token);
  final JSObject _module;
  final JSFunction _status, _release;
  final int _token;
  Future<void>? _completion;
  bool _retired = false;
  bool get isRetired => _retired;
  bool get hasSettled => _retired ||
      (_status.callAsFunction(_module, _token.toJS) as JSNumber).toDartInt != 0;
  bool get ownsCurrentModule => _module.strictEquals(
      globalContext.getProperty<JSObject?>('MinigpuModule'.toJS)).toDart;

  /// Return only this wrapper's attachment to the captured module. Completion
  /// is independent of [completed], which may still await other attachments.
  Future<void> detach() => _callContext('mgpuDestroyContext');

  Future<void> attachReplacement() {
    if (!_retired || !ownsCurrentModule) {
      return Future.error(StateError('The creation module has not retired'));
    }
    return _callContext('mgpuInitializeContext');
  }

  Future<void> _callContext(String operation) async {
    // Both operations can suspend, so they queue with every other Asyncify call.
    await _serialAsync<void>(() async {
      await _module.callMethodVarArgs<JSPromise<JSAny?>>('ccall'.toJS, [
        operation.toJS, 'void'.toJS, <JSAny>[].toJS,
        <JSAny>[].toJS, {'async': true}.toJSDeep,
      ]).toDart;
    });
  }

  Future<void> get completed => _completion ??= _wait();

  Future<void> _wait() async {
    try {
      while (true) {
        final status = (_status.callAsFunction(_module, _token.toJS)
            as JSNumber).toDartInt;
        if (status == 1) break;
        if (status != 0) {
          throw StateError('WebGPU module retirement is unconfirmed ($status)');
        }
        // This is a teardown-only observer, never a frame timer or an Asyncify
        // wait: another module call must remain free to return its attachment.
        await Future<void>.delayed(const Duration(milliseconds: 20));
      }
    } finally {
      _release.callAsFunction(_module, _token.toJS);
    }
    _retired = true;
  }
}

WebContextRetirement? mgpuObserveOwnContextRetirement(int deviceHandle) {
  const observe = '_mgpuWebObserveContextRetirement';
  const status = '_mgpuWebContextRetirementStatus';
  const release = '_mgpuWebReleaseContextObservation';
  if (!_hasMinigpuExport(observe) || !_hasMinigpuExport(status) ||
      !_hasMinigpuExport(release)) {
    return null;
  }
  final module = globalContext.getProperty<JSObject>('MinigpuModule'.toJS);
  final token = module.callMethod<JSNumber>(observe.toJS, deviceHandle.toJS).toDartInt;
  if (token <= 0) throw StateError('WebGPU creation observation was refused');
  return WebContextRetirement._(module,
      module.getProperty<JSFunction>(status.toJS),
      module.getProperty<JSFunction>(release.toJS), token);
}

/// Capture actual device features/limits for browser-selected replacement.
/// These are compatibility requirements, never a physical adapter identity.
(Set<String>, Map<String, int>) mgpuOwnDeviceRequirements(JSObject device) {
  final array = globalContext.getProperty<JSObject>('Array'.toJS);
  final object = globalContext.getProperty<JSObject>('Object'.toJS);
  final features = device.getProperty<JSObject>('features'.toJS);
  final names = array.callMethod<JSArray<JSString>>('from'.toJS, features);
  final limits = device.getProperty<JSObject>('limits'.toJS);
  final values = <String, int>{};
  JSObject? source = limits;
  for (var depth = 0; source != null && depth < 3; depth++) {
    final keys = object.callMethod<JSArray<JSString>>(
        'getOwnPropertyNames'.toJS, source);
    for (final key in keys.toDart) {
      final name = key.toDart;
      if (!name.startsWith('max') && name != 'minUniformBufferOffsetAlignment' &&
          name != 'minStorageBufferOffsetAlignment') {
        continue;
      }
      final raw = limits.getProperty<JSAny?>(key);
      if (raw == null || !raw.isA<JSNumber>()) continue;
      final value = (raw as JSNumber).toDartDouble;
      if (!value.isFinite || value < 0 || value > 9007199254740991 ||
          value != value.truncateToDouble()) {
        throw StateError('WebGPU returned an invalid $name limit');
      }
      values[name] = value.toInt();
    }
    source = object.callMethod<JSObject?>('getPrototypeOf'.toJS, source);
  }
  if (!values.containsKey('maxBufferSize') ||
      !values.containsKey('maxTextureDimension2D')) {
    throw StateError('WebGPU replacement limits are unavailable');
  }
  return ({for (final name in names.toDart) name.toDart}, values);
}

int? _positiveSafeLimit(JSObject limits, String property) {
  final value = limits.getProperty<JSAny?>(property.toJS);
  if (value == null || !value.isA<JSNumber>()) return null;
  final number = (value as JSNumber).toDartDouble;
  if (!number.isFinite || number <= 0 ||
      number > 9007199254740991 || number != number.truncateToDouble()) {
    return null;
  }
  return number.toInt();
}

/// Exact limits on the resolved GPUDevice, or null if any JS surface is absent.
(int, int)? mgpuReadOwnDeviceBufferLimits(JSObject device) {
  try {
    final limits = device.getProperty<JSAny?>('limits'.toJS);
    if (limits == null || !limits.isA<JSObject>()) return null;
    final ownLimits = limits as JSObject;
    final maxBufferSize = _positiveSafeLimit(ownLimits, 'maxBufferSize');
    final maxStorageBufferBindingSize =
        _positiveSafeLimit(ownLimits, 'maxStorageBufferBindingSize');
    if (maxBufferSize == null || maxStorageBufferBindingSize == null) {
      return null;
    }
    return (maxBufferSize, maxStorageBufferBindingSize);
  } catch (_) {
    return null;
  }
}

(int, int)? mgpuReadOwnDeviceTextureLimits(JSObject device) {
  try {
    final limits = device.getProperty<JSAny?>('limits'.toJS);
    if (limits == null || !limits.isA<JSObject>()) return null;
    final ownLimits = limits as JSObject;
    final dimension = _positiveSafeLimit(ownLimits, 'maxTextureDimension2D');
    final layers = _positiveSafeLimit(ownLimits, 'maxTextureArrayLayers');
    if (dimension == null || layers == null) return null;
    return (dimension, layers);
  } catch (_) {
    return null;
  }
}

/// The loss promise belonging to minigpu's owned device, if readable.
Future<void>? mgpuOwnDeviceLost(JSObject device) {
  try {
    final promise = device.getProperty<JSPromise<JSAny?>?>('lost'.toJS);
    if (promise == null) return null;
    return promise.toDart.then((_) {});
  } catch (_) {
    return null;
  }
}

/// Destroy the retained creation device, never an ambient/replacement device.
/// Completion of its loss promise is observed separately by the platform owner.
void mgpuDestroyOwnDevice(JSObject device) =>
    device.callMethod<JSAny?>('destroy'.toJS);

@JS('_mgpuFlushBatch')
external void _mgpuFlushBatchJs();

/// Submits any open fire-and-forget compute batch on minigpu's device.
void mgpuFlushBatch() => _mgpuFlushBatchJs();

/// Resolves only after work already submitted to minigpu's own device queue
/// has completed. The device comes from Emscripten's handle table, so a host
/// page's unrelated WebGPU device cannot acknowledge this work early.
Future<void> mgpuWaitForSubmittedWork() async {
  final object = getWebGpuJsObject(mgpuGetWGPUDeviceHandle());
  if (object == null) {
    throw StateError('minigpu has no initialized WebGPU device');
  }
  final device = object as GPUDevice;
  await (device.queue as _GPUQueue).onSubmittedWorkDone().toDart;
}

@JS('_mgpuGetWGPUBufferHandle')
external JSNumber _mgpuGetWGPUBufferHandleJs(MGPUBuffer buf);

/// Returns the Emscripten integer handle for a WGPUBuffer inside [buf].
/// Pass to [getWebGpuJsObject] from webgpu_interop.dart to get the JS GPUBuffer.
int mgpuGetWGPUBufferHandle(MGPUBuffer buf) =>
    _mgpuGetWGPUBufferHandleJs(buf).toDartDouble.toInt();

// 🔴 Scoped to `MinigpuModule`, NOT `Module`: classic Emscripten builds all
// claim `globalThis.Module`, and any other wasm on the page (miniav_web's
// audio module in the meet app) clobbers it — every `_mgpu*` extern then
// resolves on the WRONG module and dies as "func is not a function".
// `module_loader.dart` publishes the glue's private instance under this name
// (or adopts a host-page-preloaded global `Module`); nothing here may read
// the shared global.
@JS('MinigpuModule')
library minigpu_bindings;

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

Future<void> mgpuInitializeContext() async {
  // Everything else in this library assumes an initialized context, so this
  // is the one choke point where the module itself must be up first.
  await ensureMinigpuModuleLoaded();
  await ccall(
    "mgpuInitializeContext".toJS,
    "void".toJS,
    <JSAny>[].toJS,
    <JSAny>[].toJS,
    {"async": true}.toJSDeep,
  ).toDart;
}

Future<void> mgpuDestroyContext() async {
  await ccall(
    "mgpuDestroyContext".toJS,
    "void".toJS,
    <JSAny>[].toJS,
    <JSAny>[].toJS,
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
    _heapU8.setRange(start, start + bytes.length, bytes);
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
    _heapU8.setRange(start, start + bytes.length, bytes);
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
  final heap = _heapU8;
  heap.setRange(start, start + bytes.length, bytes);
  heap[start + bytes.length] = 0;
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
    heap.setRange(index, index + words.length, words);
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
    heapBytes.buffer.asFloat32List(clearStart, 5).setAll(0, clear32);

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
      heapWords.setRange(index, index + commands.length, commands);
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
      // FIX: argTypes length must match args length (4)
      ["number", "number", "number", "number"].toJS,
      [shader, groupsX.toJS, groupsY.toJS, groupsZ.toJS].toJS,
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
      ["number", "number", "number", "number"].toJS,
      [buffer, ptr, elementsToRead.toJS, elementOffset.toJS].toJS,
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
      ["number", "number", "number", "number"].toJS,
      [buffer, ptr, elementsToRead.toJS, elementOffset.toJS].toJS,
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
      ["number", "number", "number", "number"].toJS,
      [buffer, ptr, elementsToRead.toJS, elementOffset.toJS].toJS,
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
      ["number", "number", "number", "number"].toJS,
      [buffer, ptr, elementsToRead.toJS, elementOffset.toJS].toJS,
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
      ["number", "number", "number", "number"].toJS,
      [buffer, ptr, elementsToRead.toJS, elementOffset.toJS].toJS,
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
      ["number", "number", "number", "number"].toJS,
      [buffer, ptr, elementsToRead.toJS, elementOffset.toJS].toJS,
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
      ["number", "number", "number", "number"].toJS,
      [buffer, ptr, elementsToRead.toJS, elementOffset.toJS].toJS,
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
    await ccall(
      "mgpuReadSyncUint64".toJS, // Use sync C++ function
      "void".toJS,
      ["number", "number", "number", "number"].toJS,
      [buffer, ptr, elementsToRead.toJS, elementOffset.toJS].toJS,
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
      ["number", "number", "number", "number"].toJS,
      [buffer, ptr, elementsToRead.toJS, elementOffset.toJS].toJS,
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
      ["number", "number", "number", "number"].toJS,
      [buffer, ptr, elementsToRead.toJS, elementOffset.toJS].toJS,
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
  final int elementsRequested = size > 0 ? size : inputData.length;
  final int actualElements = elementsRequested <= inputData.length
      ? elementsRequested
      : inputData.length;
  final int byteSize = actualElements * Int8List.bytesPerElement;
  final JSNumber ptr = _malloc(byteSize.toJS);
  final int startIndex = ptr.toDartInt;
  try {
    if (startIndex < 0 || startIndex + actualElements > _heapU8.length) {
      throw StateError('Int8 write would exceed heap bounds');
    }
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
  final int elementsRequested = size > 0 ? size : inputData.length;
  final int actualElements = elementsRequested <= inputData.length
      ? elementsRequested
      : inputData.length;
  final int byteSize = actualElements * Int64List.bytesPerElement;
  final JSNumber ptr = _malloc(byteSize.toJS);
  final int startByteIndex = ptr.toDartInt;
  try {
    if (startByteIndex % 8 != 0) {
      throw StateError('Int64 pointer not 8-byte aligned');
    }
    final startElementIndex = startByteIndex ~/ Int64List.bytesPerElement;
    final heapInt64 = _heapU8.buffer.asInt64List();
    if (startElementIndex + actualElements > heapInt64.length) {
      throw StateError('Int64 write would exceed heap bounds');
    }
    heapInt64.setRange(
      startElementIndex,
      startElementIndex + actualElements,
      inputData.sublist(0, actualElements),
    );
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
  final int elementsRequested = size > 0 ? size : inputData.length;
  final int actualElements = elementsRequested <= inputData.length
      ? elementsRequested
      : inputData.length;
  final int byteSize = actualElements * Uint8List.bytesPerElement;
  final JSNumber ptr = _malloc(byteSize.toJS);
  final int startIndex = ptr.toDartInt;
  try {
    if (startIndex < 0 || startIndex + actualElements > _heapU8.length) {
      throw StateError('Uint8 write would exceed heap bounds');
    }
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
  final int elementsRequested = size > 0 ? size : inputData.length;
  final int actualElements = elementsRequested <= inputData.length
      ? elementsRequested
      : inputData.length;
  final int byteSize = actualElements * Uint16List.bytesPerElement;
  final JSNumber ptr = _malloc(byteSize.toJS);
  final int startByteIndex = ptr.toDartInt;
  try {
    final int startElementIndex = startByteIndex ~/ Uint16List.bytesPerElement;
    if (startElementIndex < 0 ||
        startElementIndex + actualElements > _heapU16.length) {
      throw StateError('Uint16 write would exceed heap bounds');
    }
    _heapU16.setRange(
      startElementIndex,
      startElementIndex + actualElements,
      inputData.sublist(0, actualElements),
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
  final int elementsRequested = size > 0 ? size : inputData.length;
  final int actualElements = elementsRequested <= inputData.length
      ? elementsRequested
      : inputData.length;
  final int byteSize = actualElements * Uint64List.bytesPerElement;
  final JSNumber ptr = _malloc(byteSize.toJS);
  final int startByteIndex = ptr.toDartInt;
  try {
    if (startByteIndex % 8 != 0) {
      throw StateError('Uint64 pointer not 8-byte aligned');
    }
    final startElementIndex = startByteIndex ~/ Uint64List.bytesPerElement;
    final heapUint64 = _heapU8.buffer.asUint64List();
    if (startElementIndex + actualElements > heapUint64.length) {
      throw StateError('Uint64 write would exceed heap bounds');
    }
    heapUint64.setRange(
      startElementIndex,
      startElementIndex + actualElements,
      inputData.sublist(0, actualElements),
    );
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
      ["number", "number", "number"].toJS,
      [buffer, ptr, byteSize.toJS].toJS,
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

/// Landing pipelines, keyed by geometry. Small and bounded: one entry per
/// distinct capture configuration seen this session.
final Map<String, JSObject> _landPipelines = {};

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
    await _module.callMethodVarArgs<JSPromise<JSAny?>>('ccall'.toJS, [
      operation.toJS, 'void'.toJS, <JSAny>[].toJS,
      <JSAny>[].toJS, {'async': true}.toJSDeep,
    ]).toDart;
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

/// Emscripten WebGPU helper — global object access.
///
/// This file intentionally has NO `@JS('Module')` library annotation so that
/// `@JS('WebGPU')` resolves to `globalThis.WebGPU` (the Emscripten Dawn
/// WebGPU helper), NOT to `Module.WebGPU` (undefined).
library webgpu_interop;

import 'dart:js_interop';

/// Dart interop type for the Emscripten `WebGPU` global object (defined in
/// the generated `minigpu_web.js`).  Contains `getJsObject`, `Internals`, etc.
@JS('WebGPU')
extension type _EmscriptenWebGpu._(JSObject _) implements JSObject {
  /// Returns the JS WebGPU object stored at [ptr] in
  /// `WebGPU.Internals.jsObjects`.  [ptr] is the integer handle returned by
  /// Emscripten for a WGPUDevice, WGPUBuffer, WGPUTexture, etc.
  external JSAny? getJsObject(JSNumber ptr);
}

/// The Emscripten `WebGPU` global.
@JS('WebGPU')
external _EmscriptenWebGpu get _emscriptenWebGpu;

/// Retrieve the JS WebGPU object (GPUDevice, GPUBuffer, GPUTexture…) from
/// its Emscripten integer handle.
///
/// Returns `null` if [handle] is 0 or the object is not present in the table.
JSObject? getWebGpuJsObject(int handle) {
  if (handle == 0) return null;
  try {
    final result = _emscriptenWebGpu.getJsObject(handle.toJS);
    return result is JSObject ? result : null;
  } catch (_) {
    return null;
  }
}

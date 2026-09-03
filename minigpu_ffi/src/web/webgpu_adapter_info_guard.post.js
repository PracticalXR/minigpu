// Appended to the generated minigpu_web.js by the build (CMakeLists.txt,
// `--post-js`), so it survives every `make build_weblib`.
//
// WebKit ships WebGPU without `GPUAdapter.info`, and Dawn's Emscripten port
// implements wgpuAdapterGetInfo as `fillAdapterInfoStruct(adapter.info, ...)`,
// which dereferences the result unguarded:
//
//   TypeError: undefined is not an object (evaluating 'info.subgroupMinSize')
//
// thrown out of the FIRST context init, so every encoder and decoder on the
// page is stillborn while `navigator.gpu` itself works. Seen on iPadOS Safari.
//
// This wraps the port's helper with the spec's shape — blank strings, zero
// subgroup sizes — when the adapter has no `info`. It runs in the glue's own
// scope (a classic, non-MODULARIZE build, where `WebGPU` is a script-level
// var), before the runtime initializes. minigpu_web's Dart loader installs
// the same guard on `GPUAdapter.prototype` for hosts that preload the glue
// their own way; the two are deliberately redundant.
(function () {
  if (typeof WebGPU === 'undefined' || !WebGPU ||
      typeof WebGPU.fillAdapterInfoStruct !== 'function') {
    return;
  }
  var fill = WebGPU.fillAdapterInfoStruct;
  WebGPU.fillAdapterInfoStruct = function (info, infoStruct) {
    return fill(info || {
      vendor: '', architecture: '', device: '', description: '',
      subgroupMinSize: 0, subgroupMaxSize: 0, isFallbackAdapter: false,
    }, infoStruct);
  };
})();

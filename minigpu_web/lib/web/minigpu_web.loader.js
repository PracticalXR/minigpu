if (!_minigpu) var _minigpu = {};
if (!_minigpu.loader) _minigpu.loader = {};

_minigpu.loader.load = function () {
    return new Promise(
        (resolve, reject) => {
            const minigpu_web_js = document.createElement("script");
            minigpu_web_js.src = "assets/packages/minigpu_web/web/minigpu_web.js";
            minigpu_web_js.onerror = reject;
            minigpu_web_js.onload = () => {
                // Publish the instance under the name the Dart bindings read
                // (they must not read the shared global `Module` — other
                // Emscripten wasm on the page claims that name too).
                const publish = () => { globalThis.MinigpuModule = Module; };
                if (runtimeInitialized) { publish(); resolve(); }
                Module.onRuntimeInitialized = () => { publish(); resolve(); };
            };
            document.head.append(minigpu_web_js);
        }
    );
}

if (!_minigpu) var _minigpu = {};
if (!_minigpu.loader) _minigpu.loader = {};

// One load per page: `_minigpu.loader.loading` keeps its promise (dropped if
// the load fails), and the Dart loader (module_loader.dart) awaits that promise
// instead of instantiating a second wasm whose handles would not match.
_minigpu.loader.load = function () {
    if (_minigpu.loader.loading) return _minigpu.loader.loading;
    // The Dart loader got here first: its instance is the page's.
    if (globalThis.MinigpuModule) return Promise.resolve();
    const loading = new Promise(
        (resolve, reject) => {
            const fail = (error) => {
                _minigpu.loader.loading = null;
                reject(error);
            };
            const minigpu_web_js = document.createElement("script");
            minigpu_web_js.src = "assets/packages/minigpu_web/web/minigpu_web.js";
            minigpu_web_js.onerror = fail;
            minigpu_web_js.onload = () => {
                // Publish the instance under the name the Dart bindings read
                // (they must not read the shared global `Module` — other
                // Emscripten wasm on the page claims that name too), never over
                // an instance the Dart loader already published and uses.
                const publish = () => {
                    if (!globalThis.MinigpuModule) globalThis.MinigpuModule = Module;
                };
                if (runtimeInitialized) { publish(); resolve(); }
                Module.onRuntimeInitialized = () => { publish(); resolve(); };
                // A wasm that fails to instantiate never initializes: settle
                // the load rather than leave its awaiters waiting forever.
                const prevAbort = Module.onAbort;
                Module.onAbort = (what) => {
                    if (prevAbort) prevAbort(what);
                    if (!runtimeInitialized) fail(new Error("minigpu wasm aborted: " + what));
                };
            };
            document.head.append(minigpu_web_js);
        }
    );
    _minigpu.loader.loading = loading;
    return loading;
}

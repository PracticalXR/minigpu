# minigpu

## 1.5.2

- Add buffer copy

## 1.5.1

- fix frame wait and timeouts

## 1.5.0

- Fix handle issue, release 1.5.0

## 1.4.15

- fix fallback paths

## 1.4.14

- fix Tier C

## 1.4.12

- fix texture path

## 1.4.11

- fixing build hook

## 1.4.9

- tryfix central dawn location

## 1.4.8

- fixes texture path on windows, fixes texture view on web

## 1.4.7

- Fixes logger characters

## 1.4.6

- fix garbled adapter name in logs: WGPUStringView.data is not null-terminated, copy to std::string before passing to snprintf
- fix garbled adapter name in logs: WGPUStringView.data is not null-terminated, copy to std::string before passing to snprintf

## 1.4.5

- fix Dawn built inside pub-cache instead of system dir: pass DAWN_DIR from Dart hook as cmake -D define so cmake subprocess inherits the correct path regardless of env; fix FETCHCONTENT_BASE_DIR pointing to pub-cache (now uses cmake binary dir)
- add Minigpu.setLogCallback / setLogLevel: routes native Dawn/GPU log lines through a Dart callback (NativeCallable.listener); mgpuSetLogCallback + mgpuSetLogLevel exported from C layer; all stderr calls in minigpu_external.cpp replaced with structured LOG_ERROR/INFO/WARN/DEBUG macros

## 1.4.4

- fix FormatException on non-UTF-8 bytes in setLogCallback: use Utf8Decoder(allowMalformed: true) instead of toDartString()

## 1.4.3

- improve adapter selection to prefer discrete GPU using dawn native EnumerateAdapters; fixes incorrect adapter picked on Optimus laptops

## 1.4.2

- fixes dawn library not being found

## 1.4.1

- added bindings observer

## 1.4.0

- adds minigpu_view, gpu_pipeline libraries
- adds `Minigpu.destroySync()` for use in synchronous hot-reload teardown hooks (e.g. `MinigpuBinding` from `minigpu_flutter`)

## 1.3.0

- Adds Texture Sharing

## 1.2.4-WIP

- working on texture imports

## 1.2.3

- adds VRAM API
- fixes memleaks and broken tests

## 1.2.2

- fixes memleaks and broken tests

## 1.2.1

- Fix: fresh builds need dawn find off
- Change: 1.2.0 migrates minigpu to direct webgpu usage
- Breaking: Changed setData and references to .write
- Fix: broken compute shader and buffer finalizers
## 1.2.0

## 1.1.9

- fix pubspec version issue
## 1.1.8

- fixed concurrent buffer op crash
## 1.1.6

- fixed problem with audio input capture providing raw data

## 1.1.5

- Refactored example
- Various fixes
- Updated native assets to code assets
- Memory problems fixed on web and ffi

## 1.1.3

- breaking: import package instead of buffer and shader separately
- fix: pubspec repository url
- adds: tensor package protoype
- fix: issue with reading buffer segments fixed

## 1.1.2

- fix: create dawn dir to prevent first run error.

## 1.1.1

- fix: split download command for quiet fail on remote add

## 1.1.0

- fix: dawn git not running properly

## 1.0.9

- fix: prevent using project root on ffi since pub wont see the file

## 1.0.8

- fix: pub.dev still missing project root file

## 1.0.7

- fix: project root file missing

## 1.0.6

- fix: minigpu_ffi must also use flutter in pubspec or pub.dev analysis fails
- fix: issue with project root finding as package

## 1.0.5

- fix: must have flutter in pubspec or pub.dev analysis fails

## 1.0.4

- fix: updates to readme

## 1.0.3

- fix: remove flutter from package pubspec.yaml
- fix: updates to readme

## 1.0.2

- new: explicity set supported platforms in pubspec.yaml for pub.dev

## 1.0.1

- breaking: Uses dart native assets
see updated readme.
- implements platform stub for native assets to coexist with flutter plugins.
- uses native_toolchain_cmake 0.0.4

## 1.0.0

- Initial version.

## 1.7.0

- Docs corrected: `MinigpuBinding` was described as preventing stale
  `NativeCallable` invocations during hot reload. It never could — `reassemble`
  is synchronous, so it can stop new work but cannot wait for work already
  handed to the GPU worker thread, and isolate teardown deletes callbacks
  regardless of what Dart does. That bug is fixed in minigpu_ffi, which now
  delivers completions on a Dart native port. This widget is a teardown
  ORDERING aid: it runs your registered GPU disposal at the one moment
  (hot reload) when nothing else does. No functional change.

## 1.6.1

- released 08/13/26 - MR

## 1.6.0

- Version bump to keep the minigpu 1.6.0 family aligned. No functional change
  in this package.

## 1.5.9

- Version lockstep with minigpu 1.5.9; no API change.

## 1.5.8

## 1.5.7

## 1.5.6

## 1.5.5

## 1.5.4

- fix release version pins

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

- Initial release. Re-exports `minigpu` and provides `MinigpuFlutterBinding` for registering synchronous hot-restart teardown callbacks.

## 1.9.1

- Version bump for the minigpu 1.9.1 family. No functional change in this
  package.

## 1.9.0

- Docs: a hot reload keeps every GPU object and does not re-run `gpu.init()`,
  so register callbacks that drop objects you rebuild on next use, never
  `gpu.destroySync`. A hot restart never runs these callbacks.

## 1.8.1

- Version alignment with the minigpu 1.8.1 family release.

## 1.7.0

- Docs fix: `MinigpuBinding` does not prevent stale GPU callbacks during hot
  reload, as the docs claimed; that crash is fixed in minigpu_ffi 1.7.0. The
  widget is a teardown-ordering aid: it runs your registered GPU cleanup on
  hot reload, when nothing else would. No functional change.

## 1.6.1

- released 08/13/26 - MR

## 1.6.0

- Version bump for the minigpu 1.6.0 family. No functional change in this
  package.

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

- Fix garbled adapter names in logs.

## 1.4.5

- Fix Dawn being built inside the pub cache instead of the shared Dawn
  directory.
- Add `Minigpu.setLogCallback` / `setLogLevel`: native Dawn/GPU log lines are
  routed to a Dart callback.

## 1.4.4

- Fix a `FormatException` in `setLogCallback` when a log line is not valid
  UTF-8.

## 1.4.3

- Prefer the discrete GPU when selecting an adapter; fixes the wrong adapter
  being picked on Optimus laptops.

## 1.4.2

- fixes dawn library not being found

## 1.4.1

- added bindings observer

## 1.4.0

- Initial release. Re-exports `minigpu` and provides `MinigpuFlutterBinding`
  for registering synchronous hot-restart teardown callbacks.

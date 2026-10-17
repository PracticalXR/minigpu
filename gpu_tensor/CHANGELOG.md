# gpu_tensor

## 1.9.0

- Version alignment with the minigpu 1.9.0 family release.

## 1.8.1

- Version alignment with the minigpu 1.8.1 family release.

## 1.7.0

- Version bump for the minigpu 1.7.0 family. Tensor kernels now use the
  on-disk shader cache from minigpu_ffi 1.7.0, so they compile once per
  machine instead of on every launch. No API change in this package.

## 1.6.1

- released 08/13/26 - MR

## 1.6.0

- Version bump for the minigpu 1.6.0 family; picks up the shared,
  reference-counted native context from minigpu_ffi 1.6.0. No API change in
  this package.

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

- adds minigpu_view, gpu_pipeline libraries

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

- Breaking: Changed setData and references to .write
- Change: 1.2.0 migrates minigpu to direct webgpu usage
- Fix: fresh builds need dawn find off
- Fix: broken compute shader and buffer finalizers

## 1.2.0

## 1.1.9

- fix pubspec version issue

## 1.1.8

- fixed concurrent buffer op crash

## 1.1.7

- fixed fft operations for 1D tensors
- fixed memory leaks

## 1.1.6

- fixed problem with audio input capture providing raw data

## 1.0.1

- Fix: pub.dev needs flutter dependency in pubspec.yaml

## 1.0.0

- Initial version:
  - GPU tensor operations: addition, subtraction, multiplication and division.
  - GPU-accelerated 1D and 2D FFT.
  - Tensor data operations: slice, reshape, getElement, head and tail.

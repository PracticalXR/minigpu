## 1.9.2

- Fix `DynamicStage` returning stale data when a step writes a key in place
  (`outputKey` equal to the key it read). A host write followed by a host read
  returned the old GPU contents, and a GPU write followed by a host read
  returned the old host copy; each step's result now reaches the next.
- Fix a GPU step that reads a CPU step's in-place write throwing a type error:
  the upload cast the stage's input tensor to `Tensor<Float32List>`.
- `DynamicStage` now runs a CPU step through the step's `StageExecutor`. It
  used to call the function itself, so `PersistentIsolateExecutor` and
  `isolateEnabled: true` had no effect inside a stage.
- A `Pipeline` no longer starts the process-wide `MainThreadMonitor` when it is
  constructed. The pipeline never read it, and its 5 ms timer ran for the life
  of the process, which kept a command-line program from exiting. Call
  `pipeline.threadMonitor.startMonitoring()` to use it.
- A shader step that fails to dispatch now reports why. The stage caught the
  error and called a `dispatchWorkgroups` method that no shader has, so a
  kernel that did not build surfaced as a `NoSuchMethodError`.
- Add `Pipeline.serializedConnections`, the connections of the config a
  pipeline was loaded from.
- The package no longer ships a 4.5 MB copy of an old minigpu web build under
  `web/packages/`.
- Rewrite the README against the current API; its examples now run as tests.
- Remove dead code and the unused `directed_graph` dependency, and fix every
  analyzer warning.

## 1.9.1

- Version bump for the minigpu 1.9.1 family. No functional change in this
  package.

## 1.9.0

- Version alignment with the minigpu 1.9.0 family release.

## 1.8.1

- Version alignment with the minigpu 1.8.1 family release.

## 1.7.0

- Version bump for the minigpu 1.7.0 family. Pipeline stages now use the
  on-disk shader cache from minigpu_ffi 1.7.0, so their kernels compile once
  per machine instead of on every launch. No API change in this package.

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

## 1.0.0

- Initial version.

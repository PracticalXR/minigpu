## 1.9.3

### Pipelines and stages

- `MediaStream` can now be subclassed outside this package: override the new
  `startCapture` and `stopCapture` hooks, which `start` and `stop` call. They
  replace private abstract methods that only this package could implement, and
  do nothing by default, for a stream that only pushes frames. Adds a
  dependency on `meta` for `@protected`.
- `Pipeline.start()` no longer initializes a `ResourceGovernor` or checks
  stages' `ResourceRequirements` against it. The governor reported two
  invented devices, so requirements those devices did not meet, such as
  `ResourceRequirements.mlInference()`, made `start()` throw. It also started
  a one-second metrics timer for every pipeline.
- Deprecate, for removal in 2.0, what the pipeline stores but never reads:
  `Pipeline.resourceGovernor`, which is now created only when read; the
  resource hints on stages, operations and `StageBuilder`
  (`withResourceRequirements`, `withMemoryStrategy`,
  `withRealTimeConstraints`, `withGPUAffinity`); `StageConcurrency` and
  `PipelineStage.stageConcurrency`; `MainThreadMonitor`'s
  `hasMicrotaskPressure`, `hasMemoryPressure` and `hasCpuPressure`, which are
  always false; and `synchronized`, which never locked.
- `Pipeline`'s `maxCachedShaders` and `maxCachedTensors` now size
  `resourceCache`. They were ignored, and the cache always held 100 shaders
  and 200 tensors.
- Before a stage runs, `runOnce` checks its inputs against the stage's input
  ports, and throws an `ArgumentError` when a required input is missing, an
  `ExactInputPort` gets another shape, or an input has too few or too many
  dimensions. In a running pipeline the stage fails with a `StageErrorEvent`
  instead. Input ports were not checked.
- A stage with several output ports must produce a tensor named after each
  one, or the run throws a `StateError`. An unmatched port used to get the
  last step's output, or failing that any tensor, so two ports could carry
  the same tensor. A `FixedOutputPort` now also checks that its tensor has
  the port's number of elements.
- A shader step whose output port cannot work out the output's shape throws a
  `StateError`. It used to print a warning and give the output the shape of
  its first input.
- When a `StreamMergeStage` fails to merge, a running pipeline reports a
  `StageErrorEvent` and the stage passes the first stream on unmerged, and
  `runOnce` throws the error. The stage used to print the error and pass the
  first stream on, so neither the pipeline nor the caller saw it.
- Stop printing from library code: shader steps no longer print compile
  failures they also throw, pre-allocated outputs or parameter overrides;
  `ResourceCache` no longer prints each resource it fails to destroy;
  `StreamMergeStage` no longer prints on every merge or when disposed; and
  `MainThreadMonitor` no longer prints when it starts, stops or is disposed.
- On a stream of host frames, a CPU step's output that a GPU step reads is
  uploaded in its own element type. Its bytes used to be read as float32
  values whatever its type, so an `Int32List` or `Uint8List` output reached
  the GPU as garbage.
- Host frames of `float64`, `float16`, `int64` or `uint64` data are converted
  to float32 by value. Their bytes used to be read as float32 values.
- `Pipeline.dispose()` can be called more than once; later calls complete
  with the first.

### CPU steps

- `StageBuilder.executeCPU` takes an `executor`, such as a
  `PersistentIsolateExecutor`, to run the step somewhere other than inline.
  Only a `CPUStageOperation` built by hand could take one before.
- Deprecate, for removal in 2.0, `StageBuilder.executeDynamicCPU` and
  `DynamicCPUOperation`: the package does not run Dart source, so every run
  throws `UnimplementedError`. In a JSON network a `dynamic_cpu` step is now
  a validation error, as are `merge` and `split` steps, which built a
  placeholder shader that failed when it ran.

### JSON networks

- Export the JSON network and asset API from
  `package:gpu_pipeline/gpu_pipeline.dart`: `PipelineLoader`,
  `PipelineSerializer`, `PipelineValidator` and its report, `RuntimeBuilder`,
  the config models, patches (`PatchEnvelope`, its ops, `PatchApplier` and
  `RuntimePatchExecutor`), `AssetManager`, `StaticAsset`, `AssetType`,
  `AssetEvent` and `DisposableToken`. They needed `src/` imports before.
- `PipelineLoader.loadFromJson` throws a `PipelineValidationException`, which
  carries the `ValidationReport`, when the network has a validation error,
  and builds nothing. It used to build the network anyway, so a step without
  an `operation` name failed with a null check, and an unknown `op_kind` with
  an `UnsupportedError`. Warnings are in `PipelineLoadResult.validation`.
- Validation checks much more of a network: operation kinds, the names of
  `cpu` and `tensor_op` steps against their registries, shader sources,
  expressions, queue settings, ports, asset sources and strategies, duplicate
  port and asset names, connections and their cycles, and `latest_autorun`
  links that would start each other forever. It also reports warnings, such
  as an unknown `stage_kind` or a stage asset that the network does not
  declare. `doc/serialization.md` lists every code.
- A port `shape_expr` that refers to an input, such as `same_as`, no longer
  fails validation with `SHAPE_INPUT_UNKNOWN`: validation evaluated every
  expression without inputs. It now checks that the inputs an expression
  names are ones the stage gives the calculation.
- Connections route data. Each connection passes its `from_output` to its
  `to_input`; without them it uses the only output or input, or the one
  named `output` or `input`. An input port that no connection feeds reads
  the run input of its name. Connections used to set only the order the
  stages ran in. A pipeline with connections cannot also merge streams:
  adding a stage with `StreamRouting.merged` throws a `StateError`.
- Every stage builds as a `DynamicStage` with its ports and parameters,
  whatever its `stage_kind`, and another kind is a validation warning. A
  stage of another kind without operations used to build empty, without its
  ports and parameters.
- Argument references (`param:`, `tensor:`, `shape:` and `size:`) work in
  `cpu` and `expression_cpu` steps as well as `tensor_op` steps, and see
  every tensor the stage holds. A reference to a name that is not there
  throws an `ArgumentError`. References used to resolve to `null` for an
  unknown name, and saw only the tensors with a GPU copy.
- A `tensor_op` step's `named_arguments` reach the operation, and win over
  the other keys of `arguments`. A pipeline run used to drop them.
- A `cpu` step's processor receives the keys of its `arguments` other than
  `args`, with references replaced, along with its `named_arguments`. It used
  to receive only `named_arguments`.
- Add `PatchEnvelope.fromJson`. `PatchApplier.apply` and
  `RuntimePatchExecutor.applyAndRebuild` throw an `ArgumentError` when an op
  names a pipeline, stage or parameter the network does not have. They used
  to skip an op on a missing stage, apply the rest, and add any parameter a
  `set_param` op named. `applyAndRebuild` validates the patched
  network first and throws a `PipelineValidationException` on an error,
  leaving the live pipelines as they were. `RuntimePatchExecutor.apply`
  changes nothing, and returns false, unless every op applies; it used to
  apply the ops before the first that failed.
- `PipelineLoadResult.dispose()` and `RuntimeBuildResult.dispose()` dispose
  the pipelines they built, after stopping the links between them. They used
  to stop only the links, so `PipelineLoadResult.dispose()` disposed the
  asset manager under pipelines that were never disposed.
- Fix every `exact_input` port in a JSON network failing to build with
  `Unknown input port type`: decoded JSON lists are `List<dynamic>`, and the
  builder accepted only `List<int>`.

### Assets

- An `AssetManager` counts the holders of each asset. A pipeline holds the
  assets it loads until it stops or is disposed, and an asset that nothing
  holds is removed, unless it is persistent: a network asset with
  `"cache": true`, a preloaded asset, or a per-call asset added with
  `promoteToCache: true`. A persistent asset stays cached until the cache
  limit evicts it, least recently used first. Every loaded asset used to be
  persistent, so the cache never shrank, and loading an asset again counted
  its size twice.
- `Pipeline.start()` loads only the network assets whose `strategy` is
  `preload`, the default, then each stage's assets once. An optional asset
  that fails to load emits a `failed` event and is skipped; a required one
  makes `start()` release what it loaded and throw. It used to load every
  network asset whatever its strategy, load each stage asset twice, and
  ignore a stage asset's `isRequired`.
- `AssetManager.registerPerCall` throws a `StateError`, and adds nothing, for
  a name that something holds, such as a pipeline's asset; it used to replace
  that asset. Disposing its token removes only the assets it added.
- Embedded asset data is read strictly. A `data` string is text, unless
  `type_hint` names a binary type, when it is base64; it used to be decoded
  as base64 whenever it looked like base64, so text could become bytes.
  `bytes` is base64 or a list of byte values. An unknown source type,
  malformed data and a `url` response other than 200 are errors; a stage
  asset's `url` source used to load nothing on such a response.
- Deprecate, for removal in 2.0, `StageBuilder.withGeneratedAsset`,
  `StageBuilder.withGeneratedInput` and `StaticAsset.generated`: nothing
  generates assets, so they make `start()` fail.
- On the web, `url` sources load with `fetch`; they used `dart:io`, which the
  web does not have. `file` sources throw an `UnsupportedError` there. Adds a
  dependency on `web`.
- A `Pipeline` given an `AssetManager` keeps that manager's cache limit, which
  other pipelines may share, unless `maxMemoryMB` is passed. It used to reset
  the limit to 4096 MB. Add `AssetManager.cacheLimit`.
- Fix an embedded asset written in JSON as a list of byte values
  (`"data": [1, 2, 3]`, or the same under `"bytes"`) failing to load, for the
  same reason as `exact_input` ports.

### Documentation and example

- Replace `docs/`, which described many APIs the package does not have, with
  three guides in `doc/`: pipelines from JSON, assets, and CPU steps. Every
  example in them runs as a test.
- The example's streams override the new `startCapture` and `stopCapture`.

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

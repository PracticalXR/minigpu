# Pipelines from JSON

A network document describes one or more pipelines in JSON: their stages, the
operations inside each stage, the assets they share, and links that pass one
pipeline's output to another. `PipelineLoader` parses, validates and builds it.

The examples on this page run as tests in `test/doc_examples_test.dart`, and
their `expect` lines show the results. They use two helpers:

```dart
Future<Tensor> _tensor(List<double> values) async {
  final t = await Tensor.zeros([values.length]);
  await t.write(Float32List.fromList(values));
  return t;
}

Future<List<double>> _read(Tensor t) async =>
    (await t.getData() as Float32List).toList();
```

## Load and run

This network has one pipeline with two stages: `scale` doubles its input, then
`shift` adds the stage parameter `delta`.

```json
{
  "schema_version": 1,
  "pipelines": [
    {
      "id": "chain",
      "queue": {"strategy": "latest_only"},
      "stages": [
        {
          "stage_id": "scale",
          "stage_kind": "DynamicStage",
          "operations": [
            {
              "op_kind": "tensor_op",
              "operation": "multiplyScalar",
              "input_keys": ["input"],
              "output_keys": ["output"],
              "arguments": {"args": [2.0]}
            }
          ]
        },
        {
          "stage_id": "shift",
          "stage_kind": "DynamicStage",
          "parameters": {"delta": 0.5},
          "operations": [
            {
              "op_kind": "tensor_op",
              "operation": "addScalar",
              "input_keys": ["input"],
              "output_keys": ["output"],
              "arguments": {"args": ["param:delta"]}
            }
          ]
        }
      ],
      "connections": [{"from_stage": "scale", "to_stage": "shift"}]
    }
  ]
}
```

```dart
final loaded = PipelineLoader().loadFromJson(_chainJson);
final pipeline = loaded.runtime.pipelines.single;
final out = await pipeline.runOnce({
  'input': await _tensor([1, 2, 3, 4]),
});
expect(await _read(out['output']!), [2.5, 4.5, 6.5, 8.5]);

await loaded.dispose();
```

`runOnce` runs every stage once and returns the outputs of the last stage. It
does not need `start()`, but it does not load assets either; see
[assets.md](assets.md).

`loadFromJson` throws a `PipelineValidationException` if the document has
errors; see [Validation](#validation). `loaded.dispose()` stops the links
between the pipelines, disposes the pipelines, and then disposes the asset
manager they share. Disposing a pipeline a second time does nothing.

## The document

Keys marked * are required. Other keys are ignored.

### Network

| Key | Meaning |
| --- | --- |
| `schema_version`* | Must be `1`. |
| `pipelines`* | The pipelines. |
| `assets` | Assets loaded when a pipeline starts; see [assets.md](assets.md). |
| `inter_pipeline_links` | See [Links between pipelines](#links-between-pipelines). |
| `notes` | Free text. |

### Pipeline

| Key | Meaning |
| --- | --- |
| `id`* | Unique within the network. |
| `stages`* | The stages. |
| `connections`* | How tensors pass between the stages, and may be `[]`; see [Stage order and routing](#stage-order-and-routing). |
| `queue` | `strategy` (`latest_only`, the default, or `latest_n`, `drop_on_overrun`, `batched`, `process_all`), and the positive integers `max_items_per_stream` (default 1), `max_total_items` (10) and `batch_size` (1). |
| `enable_benchmarking`, `debug_enabled` | Passed to the `Pipeline`. |

A pipeline's `parameters` and `metadata` are read but not used.

### Stage

| Key | Meaning |
| --- | --- |
| `stage_id`* | Unique within the pipeline. |
| `stage_kind`* | `"DynamicStage"`. |
| `operations` | Run in order; see [Operation kinds](#operation-kinds). |
| `parameters` | Names and initial values. Operations read them and patches change them. |
| `inputs`, `outputs` | Ports; see [Ports](#ports). |
| `assets` | Names of network assets the stage uses. The validator warns about a name the network does not declare; the list has no other effect. |

Every stage is built as a `DynamicStage` from its ports, parameters and
operations, so a stage of another kind builds the same way, with a warning.
`order`, `execution_target` and `resource` are read but not used.

### Operation

| Key | Meaning |
| --- | --- |
| `op_kind`* | See below. |
| `operation` | The operation's name, for `tensor_op` and `cpu`. |
| `input_keys`, `output_keys` | The tensors the operation reads and writes. |
| `arguments`, `named_arguments` | Depend on the kind. |
| `wgsl`, `shader_parameters`, `workgroup_size` | For `shader`. |

## Operation kinds

| `op_kind` | What it builds |
| --- | --- |
| `tensor_op` | A gpu_tensor operation named by `operation`, such as `addScalar` or `multiplyScalar`; `TensorOperationRegistry.availableOperations` lists them. `input_keys` defaults to `["input"]` and `output_keys` to `["output"]`. Each input key is processed on its own and written to the output key at the same position, or to the last one. Positional arguments go in `arguments.args`, and the other keys of `arguments` are named arguments, as are the keys of `named_arguments`, which win over them. An operation on two tensors, such as `add`, takes the second as `"args": ["tensor:<key>"]`. |
| `expression_cpu` | A per-element math expression on the CPU. `named_arguments` maps each output key to its expression, and `arguments` holds initial parameter values. `input_keys` defaults to `["input"]` and `output_keys` to the expression keys. See [cpu_steps.md](cpu_steps.md#expression-steps). |
| `cpu` | A host step from `CPUOperationRegistry`: the built-in `passthrough` or `debug_info`, or one you register; see [cpu_steps.md](cpu_steps.md#built-in-steps). The step's parameters are the stage parameters, plus the keys of `arguments` other than `args` and the keys of `named_arguments`. A stage parameter wins over an argument of the same name. |
| `shader` | A WGSL compute step from `wgsl`, with `shader_parameters` and `workgroup_size`. It reads the stage parameters, and ignores `arguments` and `named_arguments`. |
| `dynamic_cpu` | Deprecated, and a validation error: the package does not run Dart source. Use `expression_cpu`, or a `cpu` step you register. |
| `merge`, `split` | A validation error: they have no implementation. Write the step as a `shader`. |

Any other `op_kind` is a validation error too.

An expression step reads stage parameters by name. Here the stage parameter
`gain` is 2, and `offset` comes from `arguments`:

```json
{
  "schema_version": 1,
  "pipelines": [
    {
      "id": "fx",
      "queue": {"strategy": "latest_only"},
      "stages": [
        {
          "stage_id": "gain",
          "stage_kind": "DynamicStage",
          "parameters": {"gain": 2.0},
          "operations": [
            {
              "op_kind": "expression_cpu",
              "input_keys": ["input"],
              "arguments": {"offset": 1.0},
              "named_arguments": {"output": "x * gain + offset"}
            }
          ]
        }
      ],
      "connections": []
    }
  ]
}
```

```dart
final loaded = PipelineLoader().loadFromJson(_gainJson);
final pipeline = loaded.runtime.pipelines.single;
final out = await pipeline.runOnce({
  'input': await _tensor([1, 2, 3, 4]),
});
expect(await _read(out['output']!), [3, 5, 7, 9]);

await loaded.dispose();
```

### Argument references

Before a `tensor_op`, `cpu` or `expression_cpu` step runs, these strings in its
`arguments`, including inside `args`, and in its `named_arguments` are
replaced:

| Reference | Value |
| --- | --- |
| `param:<name>` | The stage parameter's current value. |
| `tensor:<key>` | The stage's tensor with that key. |
| `shape:<key>` | That tensor's shape. |
| `size:<key>` | That tensor's element count. |

A reference can name any tensor the stage holds at that point, including one a
CPU step has just written. A name the stage does not have makes the step throw
an `ArgumentError`. Shader steps take no references.

## Ports

Ports name a stage's inputs and outputs. A stage without ports takes one input
and gives one output, as described in
[Stage order and routing](#stage-order-and-routing).

| `port_type` | Use |
| --- | --- |
| `flexible_input` | An input of any shape. It is required unless `"required": false`, and `min_dimensions` and `max_dimensions` limit its number of dimensions. |
| `exact_input` | An input of the shape in `"shape_expr": {"op": "fixed", "shape": [...]}`, given as positive integers. |
| `dynamic_output` | An output. A `fixed` shape of integers makes its shape fixed. Another `shape_expr` sizes the output of a shader step that writes it, from the shader's input shapes and the stage parameters; with none, that output takes the first input's shape. |

Any other `port_type` is a validation error. Before a stage runs, `runOnce`
checks its inputs against its input ports, and throws an `ArgumentError` when
a required input is missing, an `exact_input` gets another shape, or an input
has too few or too many dimensions. In a running pipeline the stage fails with
a `StageErrorEvent` instead. A port's `shape_policy` and `size_policy` are read
but not used.

```json
{
  "schema_version": 1,
  "pipelines": [
    {
      "id": "fixed",
      "stages": [
        {
          "stage_id": "plus_one",
          "stage_kind": "DynamicStage",
          "inputs": [
            {
              "name": "input",
              "port_type": "exact_input",
              "shape_expr": {"op": "fixed", "shape": [4]}
            }
          ],
          "operations": [
            {
              "op_kind": "tensor_op",
              "operation": "addScalar",
              "input_keys": ["input"],
              "output_keys": ["output"],
              "arguments": {"args": [1.0]}
            }
          ]
        }
      ],
      "connections": []
    }
  ]
}
```

```dart
final loaded = PipelineLoader().loadFromJson(_exactInputJson);
final pipeline = loaded.runtime.pipelines.single;

final port = pipeline.getStage('plus_one')!.inputPorts.single;
expect(port.name, 'input');
expect(port.expectedShape, [4]);

final out = await pipeline.runOnce({
  'input': await _tensor([1, 2, 3, 4]),
});
expect(await _read(out['output']!), [2, 3, 4, 5]);

await loaded.dispose();
```

| `shape_expr` op | Keys | Shape |
| --- | --- | --- |
| `fixed` | `shape`: integers, or `"param:<name>"` for an integer parameter | As given. |
| `same_as` | `input` | That input's shape. |
| `reshape_flat` | `input` | That input's element count, as one dimension. |
| `concat_along` | `inputs`, `axis` (default 0) | The inputs joined along the axis. |
| `broadcast_pair` | `a`, `b` | The two shapes broadcast together. |

A shape that cannot be worked out when the shader runs, for instance two
inputs that do not broadcast, makes the step throw a `StateError`.

## Stage order and routing

A pipeline without `connections` runs its stages in the order they are listed,
and passes tensors from stage to stage by name. With connections, each stage
runs after the stages connected into it, and otherwise in the listed order, and
tensors pass only along the connections.

### Without connections

During `runOnce`, the available tensors are the run's inputs, plus each stage's
outputs once it finishes; an output replaces an earlier tensor of the same
name. Each stage receives:

- **Without input ports:** one tensor, named `input`: the available `output`,
  else `input`, else the first available tensor.
- **With input ports:** for each port, the available tensor of the same name,
  else `output`, else `input`, else the first. Every other available tensor is
  passed along under its own name.

`runOnce` returns the outputs of the last stage that gave any back.

### With connections

`{"from_stage": "a", "from_output": "x", "to_stage": "b", "to_input": "y"}`
passes stage `a`'s output `x` to stage `b` as its input `y`. `from_output` can
be left out when `a` has one output, or one named `output`. `to_input` can be
left out when `b` has one input port, or one named `input`; for a stage without
input ports it is `input`.

An input port that no connection feeds reads the run input of its name, and a
stage with neither input ports nor connections into it receives all the run
inputs. `runOnce` returns the last stage's outputs, and throws a `StateError`
when a stage did not produce an output that a connection names. A running
pipeline routes each frame the same way, and skips a stage when a stage it
reads produced nothing for that frame.

### Stage outputs

Each stage gives back:

- **Without output ports:** one tensor, named `output`: the first of the last
  operation's output keys that the stage holds, else `output`, else the highest
  `temp_<n>`, else the first tensor.
- **With one output port:** the tensor of the port's name, else the tensor a
  stage without output ports would give back.
- **With several output ports:** for each port, the tensor of its name. A port
  without one makes the stage throw a `StateError`.

A `fixed` output port also throws a `StateError` when its tensor has another
number of elements. `pipeline.getStageOutputMerged(stageId)` returns one
stage's outputs from the latest `runOnce`.

## Links between pipelines

A link hands one pipeline's output to another pipeline as an input. Both of
these forms are read:

- `{"from_pipeline": "a", "from_output": "output", "to_pipeline": "b", "to_input": "input"}`
- `{"from": {"pipeline": "a", "output": "output"}, "to": {"pipeline": "b", "input": "input"}}`

The pipelines, `from_output` and `to_input` are required, and `mode` is
optional. When the source finishes a run, the link takes its output named
`from_output` and sets it as the target's external input `to_input`; a run
without that output passes nothing on. The target's next `runOnce` uses it.
Inputs passed to `runOnce` win over external inputs of the same name, and
`pipeline.setExternalInput` and `clearExternalInputs` set them by hand.

| `mode` | Behaviour |
| --- | --- |
| `latest` | Sets the input. This is the default, and any unknown mode means `latest`. |
| `latest_autorun` | Also runs the target, if `start()` has been called on it, one run at a time, and clears its external inputs afterwards. A run that fails is reported as a `PipelineErrorEvent` on the target. |

The link reacts to an event that is delivered asynchronously, so let the event
queue run before you use the target:

```json
{
  "schema_version": 1,
  "pipelines": [
    {
      "id": "a",
      "queue": {"strategy": "latest_only"},
      "stages": [
        {
          "stage_id": "plus_one",
          "stage_kind": "DynamicStage",
          "operations": [
            {
              "op_kind": "tensor_op",
              "operation": "addScalar",
              "input_keys": ["input"],
              "output_keys": ["output"],
              "arguments": {"args": [1.0]}
            }
          ]
        }
      ],
      "connections": []
    },
    {
      "id": "b",
      "queue": {"strategy": "latest_only"},
      "stages": [
        {
          "stage_id": "times_two",
          "stage_kind": "DynamicStage",
          "operations": [
            {
              "op_kind": "tensor_op",
              "operation": "multiplyScalar",
              "input_keys": ["input"],
              "output_keys": ["output"],
              "arguments": {"args": [2.0]}
            }
          ]
        }
      ],
      "connections": []
    }
  ],
  "inter_pipeline_links": [
    {
      "from_pipeline": "a",
      "from_output": "output",
      "to_pipeline": "b",
      "to_input": "input",
      "mode": "latest"
    }
  ]
}
```

```dart
final loaded = PipelineLoader().loadFromJson(_linkJson);
final byId = {for (final p in loaded.runtime.pipelines) p.id: p};

await byId['a']!.runOnce({
  'input': await _tensor([1, 2, 3, 4]),
});
// The link runs on a's run-complete event, which is delivered
// asynchronously.
await pumpEventQueue();
final out = await byId['b']!.runOnce({});
expect(await _read(out['output']!), [4, 6, 8, 10]);

await loaded.dispose();
```

## Validation

`loadFromJson` validates the document before it builds anything. If it finds
an error, it throws a `PipelineValidationException`, whose `report` lists the
problems, and builds nothing. Warnings don't stop the build;
`loaded.validation.warnings` lists them.

To check a document without building it, validate the decoded config. This
one declares a stage twice:

```json
{
  "schema_version": 1,
  "pipelines": [
    {
      "id": "p",
      "stages": [
        {"stage_id": "s", "stage_kind": "DynamicStage"},
        {"stage_id": "s", "stage_kind": "DynamicStage"}
      ],
      "connections": []
    }
  ]
}
```

```dart
final config = NetworkConfig.decode(_duplicateStageJson);
final report = PipelineValidator().validate(config);

expect(report.ok, isFalse);
expect(report.errors.single.code, 'DUP_STAGE_ID');
expect(report.errors.single.path, 'pipelines[0].stages[1].stage_id');
```

`PipelineSerializer().parse(json)` does both in one call and returns the
`config` and its `report`.

Each `ValidationIssue` has a `code`, a `message` and a `path`, and `warning` is
true for a warning. `report.ok` is true when there are no errors, and
`report.toJson()` gives the whole report as JSON. The errors are:

| Code | Problem |
| --- | --- |
| `UNSUPPORTED_SCHEMA_VERSION` | `schema_version` is not 1. |
| `DUP_PIPELINE_ID`, `DUP_STAGE_ID`, `DUP_PORT_NAME`, `DUP_ASSET_NAME` | A name is used twice. |
| `ASSET_STRATEGY_UNKNOWN` | An asset's `strategy` is not `preload`, `lazy`, `per_call` or `stream`. |
| `ASSET_SOURCE_INVALID` | A required asset's source cannot load; see [assets.md](assets.md#assets-in-a-json-network). |
| `QUEUE_STRATEGY_UNKNOWN` | A queue `strategy` is not one of those listed under [Pipeline](#pipeline). |
| `QUEUE_INVALID` | `max_items_per_stream`, `max_total_items` or `batch_size` is not a positive integer. |
| `PORT_TYPE_UNKNOWN` | An input's `port_type` is not `flexible_input` or `exact_input`, or an output's is not `dynamic_output`. |
| `PORT_SHAPE_INVALID` | An `exact_input` has no `fixed` shape of positive integers. |
| `PORT_DIMENSIONS_INVALID` | `min_dimensions` or `max_dimensions` is negative, or the minimum is over the maximum. |
| `SHAPE_INVALID_DIM`, `SHAPE_PARAM_MISSING`, `SHAPE_EVAL_ERROR` | A `fixed` output shape has a dimension that is not a positive integer, names a parameter that is not an integer stage parameter, or has no `shape` list. |
| `SHAPE_INPUT_UNKNOWN` | Another output shape names an input its calculation does not receive. For a port that shader steps write, those are the shaders' input keys; for any other port, `input`, the stage's input ports and its operations' keys. |
| `SHAPE_RESULT_EMPTY`, `SHAPE_AXIS_OOB` | A `concat_along` shape has no `inputs`, or an `axis` that is not a non-negative integer. |
| `SHAPE_OP_UNKNOWN` | A `shape_expr` op is not one of those listed under [Ports](#ports). |
| `OP_KIND_MISSING` | An `op_kind` is empty. |
| `OP_KIND_UNSUPPORTED` | An operation is `dynamic_cpu`, `merge` or `split`. |
| `OP_KIND_UNKNOWN` | An `op_kind` is not `tensor_op`, `cpu`, `expression_cpu` or `shader`. |
| `OP_NAME_MISSING` | A `tensor_op` or `cpu` step has no `operation`. |
| `OP_NAME_UNKNOWN` | A `tensor_op` or `cpu` step names an operation that is not registered. |
| `SHADER_SOURCE_MISSING` | A `shader` step has no `wgsl`. |
| `EXPRESSION_MISSING` | An `expression_cpu` step has no `named_arguments`. |
| `EXPRESSION_INVALID` | An expression does not parse. |
| `EXPRESSION_UNKNOWN_NAME` | An expression uses a name that is not a numeric stage parameter, a key of `arguments`, or an input other than the one it reads as `x`. |
| `CONNECTION_UNKNOWN_STAGE` | A connection names a stage that doesn't exist. |
| `CONNECTION_CYCLE` | The connections form a cycle, so the stages cannot be ordered. |
| `CONNECTION_UNKNOWN_OUTPUT`, `CONNECTION_UNKNOWN_INPUT` | `from_output` names an output the stage doesn't have, or `to_input` an input port it doesn't have. |
| `CONNECTION_AMBIGUOUS_OUTPUT`, `CONNECTION_AMBIGUOUS_INPUT` | `from_output` or `to_input` is left out, and the stage has several to choose from. |
| `CONNECTION_DUPLICATE_INPUT` | Two connections feed the same input of a stage. |
| `LINK_UNKNOWN_FROM_PIPELINE`, `LINK_UNKNOWN_TO_PIPELINE` | A link names a pipeline that doesn't exist. |
| `LINK_UNKNOWN_FROM_OUTPUT` | The last stage of the source pipeline has no output named `from_output`. |
| `LINK_UNKNOWN_TO_INPUT` | No stage of the target pipeline reads a run input named `to_input`: none has an input port of that name that no connection feeds, or is a stage without input ports that no connection feeds. |
| `LINK_AUTORUN_CYCLE` | `latest_autorun` links form a loop, so each run would start the next forever. |

The warnings are:

| Code | Problem |
| --- | --- |
| `STAGE_KIND_UNKNOWN` | A `stage_kind` is not `DynamicStage`. The stage builds as one. |
| `ARGUMENTS_UNUSED` | A `shader` step has `arguments` or `named_arguments`, which shaders ignore. |
| `ASSET_SOURCE_INVALID` | An optional asset's source cannot load; `start()` skips it. |
| `ASSET_UNDECLARED` | A stage's `assets` names an asset the network doesn't declare. |
| `LINK_DUPLICATE_INPUT` | Two links feed the same input of a pipeline; the source that finishes last wins. |

## Patches

A `PatchEnvelope(pipelineId:, ops:)` changes one pipeline of a loaded network.
Its ops are:

| Op | JSON `op` | Change |
| --- | --- | --- |
| `SetParamOp(stageId:, name:, value:)` | `set_param` | Sets a stage parameter. |
| `AddOperationOp(stageId:, operation:, index:)` | `add_operation` | Inserts an operation at `index`, or appends it when `index` is null or past the end. |
| `ReplaceStageOp(targetStageId:, stage:)` | `replace_stage` | Replaces a stage. Connections follow a changed `stage_id`. |

Ops and envelopes have `toJson`, and `PatchOp.fromJson` and
`PatchEnvelope.fromJson` read them back.

`RuntimePatchExecutor` applies a patch to a map of live pipelines:

- `apply(config, patch)` handles patches made only of `set_param` ops. It sets
  the values on the live stages and returns `true`. It changes nothing and
  returns `false` when the patch holds another op, or names a pipeline, stage
  or parameter the live pipelines don't have, or a value a parameter rejects.
  It never changes the config; keep that in step with
  `PatchApplier().apply(config, patch)`.
- `applyAndRebuild(config, patch)` applies the patch to the config, validates
  the result and rebuilds every pipeline in the network on the same asset
  manager. It puts the new pipelines in the map and returns the new `config`
  and `runtime`. It throws an `ArgumentError` when the patch names something
  the network doesn't have, and a `PipelineValidationException` when the
  patched network has errors; the map is then unchanged.

```dart
final loaded = PipelineLoader().loadFromJson(_gainJson);
final old = loaded.runtime.pipelines.single;
final live = {for (final p in loaded.runtime.pipelines) p.id: p};
final executor = RuntimePatchExecutor(live);
final input = await _tensor([1, 2, 3, 4]);

final gain = PatchEnvelope(
  pipelineId: 'fx',
  ops: [SetParamOp(stageId: 'gain', name: 'gain', value: 3.0)],
);
expect(executor.apply(loaded.config, gain), isTrue);
var out = await live['fx']!.runOnce({'input': input});
expect(await _read(out['output']!), [4, 7, 10, 13]);

// apply() changes the live stage only; keep the config in step.
final config = PatchApplier().apply(loaded.config, gain);

final negate = PatchEnvelope(
  pipelineId: 'fx',
  ops: [
    AddOperationOp(
      stageId: 'gain',
      operation: OperationConfig.fromJson({
        'op_kind': 'expression_cpu',
        'input_keys': ['output'],
        'named_arguments': {'output': '-x'},
      }),
    ),
  ],
);
expect(executor.apply(config, negate), isFalse);
final rebuilt = executor.applyAndRebuild(config, negate);
expect(identical(live['fx'], old), isFalse);

out = await live['fx']!.runOnce({'input': input});
expect(await _read(out['output']!), [-4, -7, -10, -13]);

await rebuilt.runtime.dispose();
await loaded.dispose();
```

After a rebuild, the old pipelines and their links are still in
`loaded.runtime`. `rebuilt.runtime.dispose()` stops the new links and disposes
the new pipelines. `loaded.dispose()` does the same for the old ones, then
disposes the asset manager, so call it last. To drop the old pipelines while
the new ones run, call `loaded.runtime.dispose()`, which leaves the asset
manager alone.

## Saving

`config.encode()`, or `PipelineSerializer().encode(config)`, writes a config
back to JSON. The output adds `"type"` keys, which the reader ignores.

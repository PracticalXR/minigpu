# CPU steps

A CPU step runs Dart code on a stage's data on the host. A stage can mix CPU
and GPU steps: each tensor keeps a host copy and a GPU copy, and data moves
between them only when the next step needs it on the other side.

The examples on this page run as tests in `test/doc_examples_test.dart`, and
their `expect` lines show the results. `_tensor` and `_read` are the helpers
shown in [serialization.md](serialization.md).

## A CPU step

`StageBuilder.executeCPU` adds a step that runs on the calling isolate. This
one clips its input to a stage parameter:

```dart
final pipeline = Pipeline(id: 'clip')
  ..addStage(
    StageBuilder('clip')
        .withParameter(Parameter.stageFloat('limit', defaultValue: 2.0))
        .executeCPU(
          (inputs, ranks, params) async {
            final limit = params['limit'] as double;
            final x = inputs['input']! as Float32List;
            return {
              'output': Float32List.fromList([
                for (final v in x) v.clamp(-limit, limit).toDouble(),
              ]),
            };
          },
          inputKey: 'input',
          outputKey: 'output',
        )
        .build(),
  );
final input = await _tensor([-3, 1, 5]);

var out = await pipeline.runOnce({'input': input});
expect(await _read(out['output']!), [-2, 1, 2]);

pipeline.getStage('clip')!.setPropertyValue('limit', 4.0);
out = await pipeline.runOnce({'input': input});
expect(await _read(out['output']!), [-3, 1, 4]);

await pipeline.dispose();
```

The processor is a `CPUProcessor`. It receives:

- `inputs`: host arrays, keyed by the step's input keys. A key the stage
  doesn't hold is filled with its `output`, else its highest-numbered
  `temp_<n>`, else its last tensor, still under the name the step asked for. If the stage
  holds no tensors at all, the step throws.
- `ranks`: each input's shape, by key.
- `params`: the stage's current parameter values, including any set with
  `setPropertyValue`.

`inputKey` defaults to the previous step's output key, else `input`, and
`outputKey` defaults to `temp_<n>`. The input arrays belong to the stage:
don't change them, and copy anything you keep after the call.

Every array the processor returns becomes a tensor of the stage under its key,
and keeps its element type. Its shape comes from an integer array returned
under `<key>_shape`, if there is one. Otherwise it takes the shape of the
step's `input`, then `output`, when that has the same number of elements, then
of any other input with that number, else it is one-dimensional. Keys ending
in `_shape` are always read as shapes, never as data. Returning `input` with a
different shape stores the array under a new `derived_input_...` key instead.

Which tensor leaves the stage is described in
[Stage order and routing](serialization.md#stage-order-and-routing).

## On a worker isolate

A step built with `isolateEnabled: true` runs on a worker isolate, which starts
on the first run and stays up until the stage is disposed. The input and
output arrays are copied to it and back. This stage is built from a
`CPUStageOperation`; with `StageBuilder`, pass
`executor: PersistentIsolateExecutor(processor)` to `executeCPU` instead:

```dart
// Top level, so it can be sent to a worker isolate.
Future<Map<String, TypedData>> squareAll(
  Map<String, TypedData> inputs,
  Map<String, List<int>> ranks,
  Map<String, dynamic> parameters,
) async {
  final x = inputs['input']! as Float32List;
  return {
    'output': Float32List.fromList([for (final v in x) v * v]),
  };
}
```

```dart
final square = CPUStageOperation(
  operationId: 'square',
  processor: squareAll,
  isolateEnabled: true,
  inputKeys: ['input'],
  outputKeys: ['output'],
);
final pipeline = Pipeline(id: 'worker')
  ..addStage(
    DynamicStage(
      stageId: 'square',
      operations: [square],
      parameterDefinitions: const {},
    ),
  );

final out = await pipeline.runOnce({
  'input': await _tensor([1, 2, 3]),
});
expect(await _read(out['output']!), [1, 4, 9]);
expect(square.executor.backend, 'isolate');

await pipeline.dispose(); // also stops the worker isolate
```

- The processor is sent to the worker. A top-level or static function always
  can be. A closure can be if everything it captures can be sent, and the
  worker then works on a copy of that state. Capturing something that can't
  be sent fails on the first run.
- An error in the processor arrives as a `StateError` whose message starts
  with `isolate stage failed:`.
- Disposing the pipeline stops the worker, and runs still waiting fail.
- Worker isolates are native only. On the web the step's first run throws
  `UnsupportedError`, so use inline steps there.
- `CPUStageOperation(executor: ...)` and `executeCPU(executor: ...)` take any
  other `StageExecutor`, an object with a `backend` name, a `run` method that
  matches the processor's signature, and `dispose`. Without one, a step uses
  `InlineExecutor`, or the worker isolate when `isolateEnabled` is true. The
  step disposes its executor when the stage is disposed.

## Expression steps

`ExpressionCPUOperation` computes each output element from a math expression.
It needs no Dart closure, so it runs the same on native and the web, and JSON
networks use it as the `expression_cpu` kind; see
[serialization.md](serialization.md#operation-kinds). This step applies a Hann
window:

```dart
final pipeline = Pipeline(id: 'window')
  ..addStage(
    DynamicStage(
      stageId: 'hann',
      operations: [
        ExpressionCPUOperation(
          operationId: 'hann',
          expressions: {
            'output':
                'n > 1 ? x * (0.5 - 0.5 * cos(2 * pi * i / (n - 1))) : x',
          },
        ),
      ],
      parameterDefinitions: const {},
    ),
  );

final out = await pipeline.runOnce({
  'input': await _tensor([1, 1, 1, 1, 1]),
});
final window = await _read(out['output']!);
const hann = [0.0, 0.5, 1.0, 0.5, 0.0];
for (var i = 0; i < hann.length; i++) {
  expect(window[i], closeTo(hann[i], 1e-6));
}

await pipeline.dispose();
```

The expressions are compiled when the step is built, so mistakes show up then:

```dart
expect(
  () => ExpressionCPUOperation(
    operationId: 'broken',
    expressions: {'output': 'x * (1 +'},
  ),
  throwsFormatException,
);
```

`expressions` maps each output key to its expression. `inputKeys` defaults to
`['input']`, and `initialParameters` gives values that the stage's parameters
of the same name override.

The primary input is the one named `input`, else the first. It can be a
float32, float64, int32 or uint8 array; any other type throws `ArgumentError`.
Each output is a float32 array with the primary input's shape. In an
expression:

| Name | Value |
| --- | --- |
| `x` | The primary input's element. |
| `i` | The element's index. |
| `n` | The number of elements. |
| `pi`, `e`, `tau` | The constants. |
| Any other name | A numeric parameter of that name. Otherwise another input of that name, read at `i`, or 0 past its end. Otherwise the step throws `StateError` when it runs. |

| Operators | Meaning |
| --- | --- |
| `+ - * / %` | Arithmetic. |
| `- !` | Negation, and logical not. |
| `< <= > >= == !=` | Comparisons, giving 1 or 0. |
| `&& \|\|` | Logic, giving 1 or 0. |
| `c ? a : b` | Choice. |

| Arguments | Functions |
| --- | --- |
| 1 | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sqrt`, `abs`, `exp`, `log`, `floor`, `ceil`, `round`, `sign`, `fract` |
| 2 | `atan2`, `pow`, `min`, `max` |
| 3 | `clamp` |

An unknown function, or the wrong number of arguments, throws
`FormatException` when the step is built. Expression steps always run inline.

## Built-in steps

`StageBuilder.cpuOp(name)`, or `"op_kind": "cpu"` with `"operation": name` in
JSON, adds a step from `CPUOperationRegistry`. Two are built in:

- `passthrough` returns its first input as `output`.
- `debug_info` prints its inputs to standard output and returns the first as
  `output`.

`CPUOperationRegistry.register(name, processor)` adds your own processor under
a name, which a JSON network can then use; register it before the network is
loaded, since validation rejects names it doesn't know.
`CPUOperationRegistry.availableOperations` lists the names.

The processor's `params` are the stage parameters, plus the step's own:
`cpuOp(name, parameters: ...)` in Dart, and in JSON the keys of `arguments`
other than `args` and the keys of `named_arguments`, with any
[references](serialization.md#argument-references) replaced. A stage parameter
wins over a step's parameter of the same name.

## Deprecated

`StageBuilder.executeDynamicCPU`, `DynamicCPUOperation` and the JSON kind
`dynamic_cpu` are deprecated and will be removed in 2.0, because the package
does not run Dart source. `executeDynamicCPU` and `DynamicCPUOperation` build,
but every run throws `UnimplementedError`, and `executeDynamicCPU` as a
stage's first step without an `inputKey` throws while the stage is built. A
`dynamic_cpu` step is a validation error, so `loadFromJson` throws. Use
`executeCPU`, an expression step or a registered step instead.

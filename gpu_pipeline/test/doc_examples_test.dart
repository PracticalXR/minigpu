import 'dart:typed_data';

import 'package:gpu_pipeline/gpu_pipeline.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:test/test.dart';

// Every example in doc/ runs here. Change an example and its copy in the
// same edit.

Future<Tensor> _tensor(List<double> values) async {
  final t = await Tensor.zeros([values.length]);
  await t.write(Float32List.fromList(values));
  return t;
}

Future<List<double>> _read(Tensor t) async =>
    (await t.getData() as Float32List).toList();

const _chainJson = r'''
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
''';

const _gainJson = r'''
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
''';

const _linkJson = r'''
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
''';

const _exactInputJson = r'''
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
''';

const _duplicateStageJson = r'''
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
''';

const _assetsJson = r'''
{
  "schema_version": 1,
  "assets": [
    {"name": "settings", "source": {"type": "embedded", "data": "{\"gain\": 3.0}"}},
    {"name": "table", "source": {"type": "embedded", "bytes": "AQIDBA=="}},
    {"name": "steps", "source": {"type": "embedded", "data": [5, 6, 7, 8]}}
  ],
  "pipelines": [
    {
      "id": "p",
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
        }
      ],
      "connections": []
    }
  ]
}
''';

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

void main() {
  group('doc/serialization.md', () {
    test('a JSON network loads and runs', () async {
      final loaded = PipelineLoader().loadFromJson(_chainJson);
      final pipeline = loaded.runtime.pipelines.single;
      final out = await pipeline.runOnce({
        'input': await _tensor([1, 2, 3, 4]),
      });
      expect(await _read(out['output']!), [2.5, 4.5, 6.5, 8.5]);

      await loaded.dispose();
    });

    test('an expression step takes stage parameters and arguments', () async {
      final loaded = PipelineLoader().loadFromJson(_gainJson);
      final pipeline = loaded.runtime.pipelines.single;
      final out = await pipeline.runOnce({
        'input': await _tensor([1, 2, 3, 4]),
      });
      expect(await _read(out['output']!), [3, 5, 7, 9]);

      await loaded.dispose();
    });

    test('patches change a live pipeline, then rebuild it', () async {
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
    });

    test('a link hands one pipeline\'s output to another', () async {
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
    });

    // Decoded JSON lists are List<dynamic>, and the builder used to require
    // List<int>, so every exact_input port from JSON threw.
    test('an exact_input port builds from JSON', () async {
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
    });

    test('validation reports problems by code and path', () {
      final config = NetworkConfig.decode(_duplicateStageJson);
      final report = PipelineValidator().validate(config);

      expect(report.ok, isFalse);
      expect(report.errors.single.code, 'DUP_STAGE_ID');
      expect(report.errors.single.path, 'pipelines[0].stages[1].stage_id');
    });
  });

  group('doc/assets.md', () {
    test('start() preloads the network assets', () async {
      final loaded = PipelineLoader().loadFromJson(_assetsJson);
      final pipeline = loaded.runtime.pipelines.single;

      await pipeline.start();
      expect(loaded.assets.getPreloadedConfig('settings'), {'gain': 3.0});
      expect(loaded.assets.getPreloadedBinary('table'), [1, 2, 3, 4]);
      expect(loaded.assets.getPreloadedBinary('steps'), [5, 6, 7, 8]);

      await loaded.dispose();
    });

    test('a CPU step reads its stage asset', () async {
      final pipeline = Pipeline(id: 'calibrated');
      pipeline.addStage(
        StageBuilder('apply_gain')
            .withAsset(
              asset: StaticAsset.text(
                name: 'calibration',
                data: '{"gain": 3.0}',
              ),
            )
            .executeCPU(
              (inputs, ranks, params) async {
                final cfg = pipeline.assetManager.getPreloadedConfig(
                  'calibration',
                )!;
                final gain = (cfg['gain'] as num).toDouble();
                final x = inputs['input']! as Float32List;
                return {
                  'output': Float32List.fromList([for (final v in x) v * gain]),
                };
              },
              inputKey: 'input',
              outputKey: 'output',
            )
            .build(),
      );

      await pipeline.start(); // loads the stage's assets
      final out = await pipeline.runOnce({
        'input': await _tensor([1, 2]),
      });
      expect(await _read(out['output']!), [3, 6]);

      await pipeline.dispose();
    });

    test('a per-call asset leaves when its token is disposed', () async {
      final assets = AssetManager();
      final seen = <String>[];
      assets.events.listen((e) => seen.add('${e.kind} ${e.name}'));

      final token = assets.registerPerCall({
        'frame_info': StaticAsset.text(name: 'frame_info', data: '{"n": 7}'),
      });
      expect(assets.getPreloadedConfig('frame_info'), {'n': 7});

      await token.dispose();
      expect(assets.get('frame_info'), isNull);
      expect(seen, ['loaded frame_info', 'evicted frame_info']);

      await assets.dispose();
    });
  });

  group('doc/cpu_steps.md', () {
    test('an inline CPU step reads a stage parameter', () async {
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
    });

    test('a CPU step runs on a worker isolate', () async {
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
    });

    test('an expression step applies a Hann window', () async {
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
    });

    test('a syntax error throws when the step is built', () {
      expect(
        () => ExpressionCPUOperation(
          operationId: 'broken',
          expressions: {'output': 'x * (1 +'},
        ),
        throwsFormatException,
      );
    });
  });
}

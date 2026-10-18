import 'dart:convert';
import 'dart:typed_data';

import 'package:gpu_pipeline/gpu_pipeline.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:test/test.dart';

Future<Tensor> _tensor(List<double> values, [List<int>? shape]) async {
  final t = await Tensor.zeros(shape ?? [values.length]);
  await t.write(Float32List.fromList(values));
  return t;
}

Future<List<double>> _read(Tensor t) async =>
    (await t.getData() as Float32List).toList();

/// JSON for a network of one pipeline, `p`.
String _network(
  List<Map<String, Object?>> stages, [
  List<Map<String, Object?>> connections = const [],
]) => jsonEncode({
  'schema_version': 1,
  'pipelines': [
    {'id': 'p', 'stages': stages, 'connections': connections},
  ],
});

Map<String, Object?> _tensorOp(
  String operation,
  String input,
  String output, {
  Map<String, Object?> arguments = const {},
  Map<String, Object?>? named,
}) => {
  'op_kind': 'tensor_op',
  'operation': operation,
  'input_keys': [input],
  'output_keys': [output],
  'arguments': arguments,
  if (named != null) 'named_arguments': named,
};

/// Loads [json], runs its pipeline once on [input] and reads the outputs.
Future<Map<String, List<double>>> _run(String json, Tensor input) async {
  final loaded = PipelineLoader().loadFromJson(json);
  final pipeline = loaded.runtime.pipelines.single;
  try {
    final out = await pipeline.runOnce({'input': input});
    return {for (final e in out.entries) e.key: await _read(e.value)};
  } finally {
    await pipeline.dispose();
    await loaded.dispose();
  }
}

/// The errors [PipelineValidator] finds in [json], as `code path`.
List<String> _errors(String json) => [
  for (final e
      in PipelineValidator().validate(NetworkConfig.decode(json)).errors)
    '${e.code} ${e.path}',
];

final _split = <String, Object?>{
  'stage_id': 'split',
  'stage_kind': 'DynamicStage',
  'outputs': [
    {'name': 'doubled', 'port_type': 'dynamic_output'},
    {'name': 'shifted', 'port_type': 'dynamic_output'},
  ],
  'operations': [
    _tensorOp(
      'multiplyScalar',
      'input',
      'doubled',
      arguments: {
        'args': [2.0],
      },
    ),
    _tensorOp(
      'addScalar',
      'input',
      'shifted',
      arguments: {
        'args': [1.0],
      },
    ),
  ],
};

final _scale = <String, Object?>{
  'stage_id': 'scale',
  'stage_kind': 'DynamicStage',
  'inputs': [
    {'name': 'x', 'port_type': 'flexible_input'},
  ],
  'operations': [
    _tensorOp(
      'multiplyScalar',
      'x',
      'output',
      arguments: {
        'args': [10.0],
      },
    ),
  ],
};

void main() {
  test('a network with errors throws instead of building', () {
    final json = _network([
      {'stage_id': 's', 'stage_kind': 'DynamicStage'},
      {'stage_id': 's', 'stage_kind': 'DynamicStage'},
    ]);
    expect(
      () => PipelineLoader().loadFromJson(json),
      throwsA(
        isA<PipelineValidationException>().having(
          (e) => [for (final i in e.report.errors) i.code],
          'error codes',
          ['DUP_STAGE_ID'],
        ),
      ),
    );
  });

  group('connections', () {
    test(
      'a connection feeds the output it names to the input it names',
      () async {
        final out = await _run(
          _network(
            [_split, _scale],
            [
              {
                'from_stage': 'split',
                'from_output': 'shifted',
                'to_stage': 'scale',
                'to_input': 'x',
              },
            ],
          ),
          await _tensor([1, 2]),
        );
        expect(out, {
          'output': [20, 30],
        });
      },
    );

    test('the validator checks the output and input a connection names', () {
      expect(
        _errors(
          _network(
            [_split, _scale],
            [
              {'from_stage': 'split', 'to_stage': 'scale'},
            ],
          ),
        ),
        ['CONNECTION_AMBIGUOUS_OUTPUT pipelines[0].connections[0]'],
      );
      expect(
        _errors(
          _network(
            [_split, _scale],
            [
              {
                'from_stage': 'split',
                'from_output': 'tripled',
                'to_stage': 'scale',
                'to_input': 'y',
              },
            ],
          ),
        ),
        [
          'CONNECTION_UNKNOWN_OUTPUT pipelines[0].connections[0].from_output',
          'CONNECTION_UNKNOWN_INPUT pipelines[0].connections[0].to_input',
        ],
      );
    });
  });

  group('ports', () {
    test('each of several output ports needs a tensor of its name', () async {
      final json = _network([
        {
          'stage_id': 'pair',
          'stage_kind': 'DynamicStage',
          'outputs': [
            {'name': 'a', 'port_type': 'dynamic_output'},
            {'name': 'b', 'port_type': 'dynamic_output'},
          ],
          'operations': [
            _tensorOp(
              'multiplyScalar',
              'input',
              'a',
              arguments: {
                'args': [2.0],
              },
            ),
            _tensorOp(
              'addScalar',
              'input',
              'bb',
              arguments: {
                'args': [1.0],
              },
            ),
          ],
        },
      ]);
      await expectLater(_run(json, await _tensor([1, 2])), throwsStateError);
    });

    test('a single output port outputs the final tensor', () async {
      final json = _network([
        {
          'stage_id': 'one',
          'stage_kind': 'DynamicStage',
          'outputs': [
            {'name': 'result', 'port_type': 'dynamic_output'},
          ],
          'operations': [
            _tensorOp(
              'addScalar',
              'input',
              'output',
              arguments: {
                'args': [1.0],
              },
            ),
          ],
        },
      ]);
      expect(await _run(json, await _tensor([1, 2])), {
        'result': [2, 3],
      });
    });

    test('an exact_input port rejects a tensor of another shape', () async {
      final json = _network([
        {
          'stage_id': 'fixed',
          'stage_kind': 'DynamicStage',
          'inputs': [
            {
              'name': 'input',
              'port_type': 'exact_input',
              'shape_expr': {
                'op': 'fixed',
                'shape': [4],
              },
            },
          ],
          'operations': [
            _tensorOp(
              'addScalar',
              'input',
              'output',
              arguments: {
                'args': [1.0],
              },
            ),
          ],
        },
      ]);
      await expectLater(
        _run(json, await _tensor([1, 2, 3])),
        throwsArgumentError,
      );
    });
  });

  group('arguments', () {
    test(
      'tensor_op named_arguments win over the other argument keys',
      () async {
        Map<String, Object?> sum(Map<String, Object?>? named) => {
          'stage_id': 'sum',
          'stage_kind': 'DynamicStage',
          'operations': [
            _tensorOp(
              'sum',
              'input',
              'output',
              arguments: {'axis': 0},
              named: named,
            ),
          ],
        };
        final input = await _tensor([1, 2, 3, 4], [2, 2]);

        expect(await _run(_network([sum(null)]), input), {
          'output': [4, 6],
        });
        expect(
          await _run(
            _network([
              sum({'axis': 1}),
            ]),
            input,
          ),
          {
            'output': [3, 7],
          },
        );
      },
    );

    test(
      'CPU arguments resolve references to any tensor of the stage',
      () async {
        Map<String, Object?> refs(Object offset) => {
          'stage_id': 'refs',
          'stage_kind': 'DynamicStage',
          'parameters': {'shift': 0.5},
          'operations': [
            _tensorOp(
              'multiplyScalar',
              'input',
              'doubled',
              arguments: {
                'args': [2.0],
              },
            ),
            {
              'op_kind': 'expression_cpu',
              'input_keys': ['doubled'],
              'arguments': {'n': 'size:doubled', 'offset': offset},
              'named_arguments': {'output': 'x + n + offset'},
            },
          ],
        };
        final input = await _tensor([1, 2, 3]);

        expect(await _run(_network([refs('param:shift')]), input), {
          'output': [5.5, 7.5, 9.5],
        });
        await expectLater(
          _run(_network([refs('param:missing')]), input),
          throwsArgumentError,
        );
      },
    );
  });

  group('patches', () {
    const gain = r'''
{
  "schema_version": 1,
  "pipelines": [
    {
      "id": "fx",
      "stages": [
        {
          "stage_id": "gain",
          "stage_kind": "DynamicStage",
          "parameters": {"gain": 2.0},
          "operations": [
            {
              "op_kind": "expression_cpu",
              "input_keys": ["input"],
              "named_arguments": {"output": "x * gain"}
            }
          ]
        }
      ],
      "connections": []
    }
  ]
}
''';

    test('a patch survives a round trip through JSON', () {
      final patch = PatchEnvelope(
        pipelineId: 'fx',
        ops: [
          SetParamOp(stageId: 'gain', name: 'gain', value: 3.0),
          AddOperationOp(
            stageId: 'gain',
            index: 0,
            operation: OperationConfig.fromJson({
              'op_kind': 'expression_cpu',
              'input_keys': ['input'],
              'named_arguments': {'output': '-x'},
            }),
          ),
          ReplaceStageOp(
            targetStageId: 'gain',
            stage: StageConfig.fromJson({
              'stage_id': 'louder',
              'stage_kind': 'DynamicStage',
              'parameters': {'gain': 4.0},
            }),
          ),
        ],
      );
      final json = jsonEncode(patch.toJson());
      final copy = PatchEnvelope.fromJson(
        jsonDecode(json) as Map<String, dynamic>,
      );
      expect(copy.pipelineId, 'fx');
      expect(copy.ops.map((o) => o.runtimeType), [
        SetParamOp,
        AddOperationOp,
        ReplaceStageOp,
      ]);
      expect(jsonEncode(copy.toJson()), json);
    });

    test('a patch that fails changes nothing', () async {
      final loaded = PipelineLoader().loadFromJson(gain);
      final pipeline = loaded.runtime.pipelines.single;
      final live = {pipeline.id: pipeline};
      final executor = RuntimePatchExecutor(live);
      final input = await _tensor([1, 2]);

      // The second op names a parameter the stage does not have.
      expect(
        executor.apply(
          loaded.config,
          PatchEnvelope(
            pipelineId: 'fx',
            ops: [
              SetParamOp(stageId: 'gain', name: 'gain', value: 3.0),
              SetParamOp(stageId: 'gain', name: 'missing', value: 1.0),
            ],
          ),
        ),
        isFalse,
      );
      expect(
        await _read((await pipeline.runOnce({'input': input}))['output']!),
        [2, 4],
      );

      expect(
        () => executor.applyAndRebuild(
          loaded.config,
          PatchEnvelope(
            pipelineId: 'fx',
            ops: [
              SetParamOp(stageId: 'gain', name: 'gain', value: 3.0),
              SetParamOp(stageId: 'other', name: 'gain', value: 1.0),
            ],
          ),
        ),
        throwsArgumentError,
      );
      expect(
        () => executor.applyAndRebuild(
          loaded.config,
          PatchEnvelope(
            pipelineId: 'fx',
            ops: [
              AddOperationOp(
                stageId: 'gain',
                operation: OperationConfig.fromJson({
                  'op_kind': 'expression_cpu',
                  'input_keys': ['output'],
                }),
              ),
            ],
          ),
        ),
        throwsA(isA<PipelineValidationException>()),
      );
      expect(identical(live['fx'], pipeline), isTrue);
      expect(live, hasLength(1));

      await pipeline.dispose();
      await loaded.dispose();
    });
  });
}

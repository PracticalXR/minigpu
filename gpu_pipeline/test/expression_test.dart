/// Unit tests for the expression engine behind ExpressionCPUOperation — the
/// web+native replacement for the removed dart_eval dynamic CPU stage.
library;

import 'dart:math' as math;
import 'dart:typed_data';
import 'package:test/test.dart';
import 'package:gpu_pipeline/gpu_pipeline.dart';
import 'package:gpu_pipeline/src/serialization/operation_factory.dart';

double eval(
  String src, {
  double x = 0,
  double i = 0,
  double n = 0,
  Map<String, double> params = const {},
  Map<String, Float32List> arrays = const {},
}) {
  final scope = ExprScope()
    ..x = x
    ..i = i
    ..n = n
    ..params = params
    ..arrays = arrays;
  return CompiledExpression.compile(src).evalAt(scope);
}

void main() {
  group('parser + evaluation', () {
    test('arithmetic precedence and parentheses', () {
      expect(eval('1 + 2 * 3'), 7);
      expect(eval('(1 + 2) * 3'), 9);
      expect(eval('10 - 4 - 3'), 3); // left assoc
      expect(eval('7 % 4 / 2'), closeTo(1.5, 1e-12));
      expect(eval('-3 * -2'), 6);
    });

    test('comparisons and logic yield 0/1 doubles', () {
      expect(eval('2 > 1'), 1);
      expect(eval('2 < 1'), 0);
      expect(eval('2 >= 2 && 1 != 0'), 1);
      expect(eval('0 || 3'), 1);
      expect(eval('!5'), 0);
      expect(eval('!0'), 1);
    });

    test('ternary', () {
      expect(eval('x > 0 ? 10 : 20', x: 1), 10);
      expect(eval('x > 0 ? 10 : 20', x: -1), 20);
      // Nested.
      expect(eval('x > 1 ? 1 : x > 0 ? 2 : 3', x: 0.5), 2);
    });

    test('functions and constants', () {
      expect(eval('sin(0)'), 0);
      expect(eval('cos(0)'), 1);
      expect(eval('sqrt(9)'), 3);
      expect(eval('pow(2, 10)'), 1024);
      expect(eval('min(3, max(1, 2))'), 2);
      expect(eval('clamp(5, 0, 2)'), 2);
      expect(eval('abs(-4.5)'), 4.5);
      expect(eval('fract(1.75)'), closeTo(0.75, 1e-12));
      expect(eval('pi'), closeTo(math.pi, 1e-12));
      expect(eval('tau / 2'), closeTo(math.pi, 1e-12));
    });

    test('variables, parameters and sibling arrays', () {
      expect(eval('x * 2 + i', x: 3, i: 4), 10);
      expect(eval('x * gain', x: 2, params: {'gain': 5}), 10);
      expect(
        eval(
          'x + other',
          x: 1,
          i: 1,
          arrays: {
            'other': Float32List.fromList([10, 20, 30]),
          },
        ),
        21,
      );
    });

    test('unknown identifier throws at EVAL with a clear message', () {
      final compiled = CompiledExpression.compile('x * mystery');
      expect(compiled.identifiers, contains('mystery'));
      expect(
        () => compiled.evalAt(ExprScope()..x = 1),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            contains('mystery'),
          ),
        ),
      );
    });

    test('syntax errors throw FormatException at COMPILE time', () {
      expect(() => CompiledExpression.compile('1 +'), throwsFormatException);
      expect(() => CompiledExpression.compile('foo('), throwsFormatException);
      expect(
        () => CompiledExpression.compile('sin(1, 2)'),
        throwsFormatException,
      );
      expect(() => CompiledExpression.compile('1 ? 2'), throwsFormatException);
      expect(() => CompiledExpression.compile('@'), throwsFormatException);
    });
  });

  group('vectorized apply', () {
    test('Hann window matches the reference formula', () {
      const expr = 'n > 1 ? x * (0.5 - 0.5 * cos(2 * pi * i / (n - 1))) : x';
      final compiled = CompiledExpression.compile(expr);

      const n = 64;
      final input = Float32List(n)..fillRange(0, n, 1.0);
      final out = Float32List(n);
      final scope = ExprScope()..n = n.toDouble();
      compiled.evalVectorized(input, out, scope);

      for (var k = 0; k < n; k++) {
        final expected = 0.5 - 0.5 * math.cos(2 * math.pi * k / (n - 1));
        expect(out[k], closeTo(expected, 1e-6), reason: 'Hann mismatch at $k');
      }
    });

    test('single-element guard avoids the n-1 division', () {
      const expr = 'n > 1 ? x * (0.5 - 0.5 * cos(2 * pi * i / (n - 1))) : x';
      final compiled = CompiledExpression.compile(expr);
      final input = Float32List.fromList([7.0]);
      final out = Float32List(1);
      compiled.evalVectorized(input, out, ExprScope()..n = 1);
      expect(out[0], 7.0);
    });
  });

  group('ExpressionCPUOperation', () {
    test(
      'processor evaluates all outputs with params + shape sidecar',
      () async {
        final op = ExpressionCPUOperation(
          operationId: 'expr_test',
          expressions: {
            'scaled': 'x * gain + bias',
            'gate': 'x > thresh ? x : 0',
          },
          initialParameters: {'gain': 2.0, 'bias': 1.0, 'thresh': 2.5},
        );

        final outputs = await op.executor.run(
          {
            'input': Float32List.fromList([1, 2, 3, 4]),
          },
          {
            'input': [4],
          },
          {},
        );

        expect(outputs['scaled'], [3, 5, 7, 9]);
        expect(outputs['gate'], [0, 0, 3, 4]);
        expect(outputs['scaled__shape'], [4]);
      },
    );

    test('runtime parameters override initial ones', () async {
      final op = ExpressionCPUOperation(
        operationId: 'expr_override',
        expressions: {'out': 'x * gain'},
        initialParameters: {'gain': 2.0},
      );
      final outputs = await op.executor.run(
        {
          'input': Float32List.fromList([1, 2]),
        },
        {
          'input': [2],
        },
        {'gain': 10.0},
      );
      expect(outputs['out'], [10, 20]);
    });

    test('bad expression fails at CONSTRUCTION, not per frame', () {
      expect(
        () => ExpressionCPUOperation(
          operationId: 'expr_bad',
          expressions: {'out': 'x **'},
        ),
        throwsFormatException,
      );
    });

    test('OperationFactory builds it from opKind expression_cpu', () {
      final op = OperationFactory.create(
        OperationConfig(
          opKind: 'expression_cpu',
          inputKeys: ['input'],
          outputKeys: ['windowed'],
          namedArguments: {'windowed': 'x * 0.5'},
        ),
      );
      expect(op, isA<ExpressionCPUOperation>());
      expect((op as ExpressionCPUOperation).expressions['windowed'], 'x * 0.5');
    });
  });
}

// lib/src/operation/expression_cpu_operation.dart
//
// The web+native dynamic CPU stage: per-element math expressions compiled
// ONCE at construction (see expression.dart) and applied in a vectorized
// loop each frame.  This replaces the removed dart_eval-based
// DynamicCPUOperation for its real-world uses (windowing, gain, thresholds,
// mixes) with zero dependencies and full web support.
import 'dart:typed_data';

import 'cpu_stage_operation.dart';
import 'expression.dart';

/// A CPU operation where each output key is produced by evaluating a math
/// expression per element of the primary input.
///
/// Expression variables:
///   `x`      — the primary input's current element
///   `i`      — element index · `n` — element count
///   `<name>` — a stage parameter, or a sibling INPUT array read at index i
///
/// Example — Hann window with a single-sample guard:
/// ```dart
/// ExpressionCPUOperation(
///   operationId: 'window',
///   inputKeys: ['input'],
///   expressions: {
///     'windowed': 'n > 1 ? x * (0.5 - 0.5 * cos(2 * pi * i / (n - 1))) : x',
///   },
/// )
/// ```
///
/// Serialization: `opKind: 'expression_cpu'` with the expressions map in
/// `namedArguments` (see OperationFactory) — fully JSON round-trippable.
class ExpressionCPUOperation extends CPUStageOperation {
  factory ExpressionCPUOperation({
    required String operationId,
    required Map<String, String> expressions,
    Map<String, dynamic> initialParameters = const {},
    List<String> inputKeys = const ['input'],
    List<String>? outputKeys,
  }) {
    // Parse errors surface HERE, at stage build — never mid-frame.
    final compiled = <String, CompiledExpression>{
      for (final e in expressions.entries)
        e.key: CompiledExpression.compile(e.value),
    };
    return ExpressionCPUOperation._(
      operationId: operationId,
      expressions: Map.unmodifiable(expressions),
      compiledExpressions: compiled,
      // The processor closes over the compiled expressions + scratch
      // buffers — inherently inline-only (isolateEnabled stays false).
      processor: _makeProcessor(compiled, initialParameters),
      inputKeys: inputKeys,
      outputKeys: outputKeys ?? expressions.keys.toList(),
      // Resolved by the stage before each run (`param:` references), and
      // passed to the processor under the stage parameters.
      arguments: initialParameters,
    );
  }

  ExpressionCPUOperation._({
    required super.operationId,
    required this.expressions,
    required this.compiledExpressions,
    required super.processor,
    required super.inputKeys,
    required super.outputKeys,
    super.arguments,
  });

  /// The raw expression sources (for serialization).
  final Map<String, String> expressions;

  final Map<String, CompiledExpression> compiledExpressions;

  static CPUProcessor _makeProcessor(
    Map<String, CompiledExpression> compiled,
    Map<String, dynamic> initialParameters,
  ) {
    // Reused across frames: output buffers per key + one scope object.
    final outScratch = <String, Float32List>{};
    final scope = ExprScope();

    return (inputs, ranks, runtimeParams) async {
      // Primary input drives the element loop.
      final primaryKey = inputs.containsKey('input')
          ? 'input'
          : (inputs.isNotEmpty ? inputs.keys.first : null);
      if (primaryKey == null) {
        throw StateError('ExpressionCPUOperation received no inputs');
      }
      final primary = _asFloat32(inputs[primaryKey]!);

      // Bind numeric parameters (initial overridden by runtime) and sibling
      // input arrays ONCE per frame — the loop only touches x and i.
      final params = <String, double>{};
      void bindParams(Map<String, dynamic> src) {
        for (final e in src.entries) {
          final v = e.value;
          if (v is num) params[e.key] = v.toDouble();
        }
      }

      bindParams(initialParameters);
      bindParams(runtimeParams);

      final arrays = <String, Float32List>{};
      for (final e in inputs.entries) {
        if (e.key == primaryKey) continue;
        arrays[e.key] = _asFloat32(e.value);
      }

      scope
        ..n = primary.length.toDouble()
        ..params = params
        ..arrays = arrays;

      final outputs = <String, TypedData>{};
      for (final entry in compiled.entries) {
        var out = outScratch[entry.key];
        if (out == null || out.length != primary.length) {
          out = Float32List(primary.length);
          outScratch[entry.key] = out;
        }
        entry.value.evalVectorized(primary, out, scope);
        outputs[entry.key] = out;
        // Same element count as the primary input — preserve its shape.
        final shape = ranks[primaryKey];
        if (shape != null) {
          outputs['${entry.key}__shape'] = Int32List.fromList(shape);
        }
      }
      return outputs;
    };
  }

  static Float32List _asFloat32(TypedData data) {
    if (data is Float32List) return data;
    if (data is Float64List) {
      final out = Float32List(data.length);
      for (var i = 0; i < data.length; i++) {
        out[i] = data[i];
      }
      return out;
    }
    if (data is Int32List) {
      final out = Float32List(data.length);
      for (var i = 0; i < data.length; i++) {
        out[i] = data[i].toDouble();
      }
      return out;
    }
    if (data is Uint8List) {
      final out = Float32List(data.length);
      for (var i = 0; i < data.length; i++) {
        out[i] = data[i].toDouble();
      }
      return out;
    }
    throw ArgumentError(
      'ExpressionCPUOperation supports Float32/Float64/Int32/Uint8 inputs, '
      'got ${data.runtimeType}',
    );
  }
}

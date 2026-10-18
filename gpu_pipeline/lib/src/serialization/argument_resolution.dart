import 'package:gpu_tensor/gpu_tensor.dart';

/// Supported reference kinds in serialized args.
enum ArgRefKind { param, tensor, shape, size, literal }

class ResolvedArguments {
  final List<dynamic> positional;
  final Map<String, dynamic> named;
  ResolvedArguments(this.positional, this.named);
}

class ArgumentResolver {
  const ArgumentResolver();

  bool _isRef(String v) =>
      v.startsWith('param:') ||
      v.startsWith('tensor:') ||
      v.startsWith('shape:') ||
      v.startsWith('size:');

  dynamic _resolveOne(
    dynamic value, {
    required Map<String, dynamic> params,
    required Map<String, Tensor> tensors,
  }) {
    if (value is String && _isRef(value)) {
      if (value.startsWith('param:')) {
        final k = value.substring(6);
        return params[k];
      }
      if (value.startsWith('tensor:')) {
        final k = value.substring(7);
        return tensors[k];
      }
      if (value.startsWith('shape:')) {
        final k = value.substring(6);
        return tensors[k]?.shape;
      }
      if (value.startsWith('size:')) {
        final k = value.substring(5);
        final t = tensors[k];
        return t == null ? null : t.shape.fold<int>(1, (a, b) => a * b);
      }
    }
    return value;
  }

  ResolvedArguments resolve(
    Map<String, dynamic> rawArgs, {
    required Map<String, dynamic> params,
    required Map<String, Tensor> tensors,
  }) {
    final pos = <dynamic>[];
    final named = <String, dynamic>{};

    // Convention: if rawArgs contains 'args' as List => positional list
    final listPart = rawArgs['args'];
    if (listPart is List) {
      for (final v in listPart) {
        pos.add(_resolveOne(v, params: params, tensors: tensors));
      }
    }
    // Remaining keys (excluding 'args') are named
    rawArgs.forEach((k, v) {
      if (k == 'args') return;
      named[k] = _resolveOne(v, params: params, tensors: tensors);
    });

    return ResolvedArguments(pos, named);
  }
}

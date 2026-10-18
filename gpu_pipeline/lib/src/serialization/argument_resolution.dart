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

  dynamic _resolveOne(
    dynamic value, {
    required Map<String, dynamic> params,
    required Map<String, Tensor> tensors,
    required Map<String, List<int>> shapes,
  }) {
    if (value is! String) return value;
    if (value.startsWith('param:')) {
      final k = value.substring(6);
      if (!params.containsKey(k)) {
        throw ArgumentError('$value names no stage parameter');
      }
      return params[k];
    }
    if (value.startsWith('tensor:')) {
      final k = value.substring(7);
      return tensors[k] ?? (throw ArgumentError('$value names no tensor'));
    }
    if (value.startsWith('shape:')) {
      return _shape(value, value.substring(6), tensors, shapes);
    }
    if (value.startsWith('size:')) {
      final shape = _shape(value, value.substring(5), tensors, shapes);
      return shape.fold<int>(1, (a, b) => a * b);
    }
    return value;
  }

  List<int> _shape(
    String ref,
    String name,
    Map<String, Tensor> tensors,
    Map<String, List<int>> shapes,
  ) =>
      shapes[name] ??
      tensors[name]?.shape ??
      (throw ArgumentError('$ref names no tensor'));

  /// Resolves the references in [rawArgs] and [named]: `param:<name>` to the
  /// value of that entry of [params], `tensor:<name>` to that entry of
  /// [tensors], and `shape:<name>` and `size:<name>` to the shape and element
  /// count of that tensor, from [shapes] or else [tensors].
  ///
  /// The positional arguments are the `args` list of [rawArgs]. The named
  /// arguments are its other keys, then [named], whose keys win.
  ///
  /// Throws an [ArgumentError] for a reference to a parameter or tensor that
  /// does not exist.
  ResolvedArguments resolve(
    Map<String, dynamic> rawArgs, {
    Map<String, dynamic> named = const {},
    required Map<String, dynamic> params,
    Map<String, Tensor> tensors = const {},
    Map<String, List<int>> shapes = const {},
  }) {
    dynamic one(dynamic v) =>
        _resolveOne(v, params: params, tensors: tensors, shapes: shapes);

    final pos = <dynamic>[];
    final listPart = rawArgs['args'];
    if (listPart is List) {
      for (final v in listPart) {
        pos.add(one(v));
      }
    }
    final resolvedNamed = <String, dynamic>{
      for (final e in rawArgs.entries)
        if (e.key != 'args') e.key: one(e.value),
      for (final e in named.entries) e.key: one(e.value),
    };
    return ResolvedArguments(pos, resolvedNamed);
  }

  /// The names that `tensor:` references in [rawArgs] and [named] point to.
  Set<String> tensorNames(
    Map<String, dynamic> rawArgs, [
    Map<String, dynamic> named = const {},
  ]) {
    final listPart = rawArgs['args'];
    return {
      for (final v in [
        if (listPart is List) ...listPart,
        for (final e in rawArgs.entries)
          if (e.key != 'args') e.value,
        ...named.values,
      ])
        if (v is String && v.startsWith('tensor:')) v.substring(7),
    };
  }
}

class ShapeEvalContext {
  final Map<String, List<int>> inputShapes;
  final Map<String, dynamic> parameters;

  ShapeEvalContext({required this.inputShapes, required this.parameters});

  dynamic resolveParam(String name) => parameters[name];
  List<int>? shapeOf(String input) => inputShapes[input];
}

class ShapeEvalError implements Exception {
  final String code;
  final String message;
  final String path;
  ShapeEvalError(this.code, this.message, this.path);
  @override
  String toString() => 'ShapeEvalError($code): $message @ $path';
}

class ShapeExprEvaluator {
  List<int> evaluate(
    Map<String, dynamic> expr,
    ShapeEvalContext ctx, {
    String path = 'shape_expr',
  }) {
    final op = expr['op'];
    switch (op) {
      case 'fixed':
        final shape = (expr['shape'] as List)
            .map((e) => _asDim(e, ctx, '$path.shape'))
            .toList();
        _validateDims(shape, path);
        return shape;
      case 'same_as':
        final input = expr['input'];
        final s =
            ctx.shapeOf(input) ??
            _error('SHAPE_INPUT_UNKNOWN', 'Unknown input $input', path);
        return List<int>.from(s);
      case 'concat_along':
        return _concatAlong(expr, ctx, path);
      case 'broadcast_pair':
        return _broadcastPair(expr, ctx, path);
      case 'reshape_flat':
        final input = expr['input'];
        final s =
            ctx.shapeOf(input) ??
            _error('SHAPE_INPUT_UNKNOWN', 'Unknown input $input', path);
        final prod = s.fold<int>(1, (a, b) => a * b);
        return [prod];
      default:
        _error('SHAPE_OP_UNKNOWN', 'Unknown op $op', path);
    }
  }

  int _asDim(dynamic v, ShapeEvalContext ctx, String path) {
    if (v is int) return v;
    if (v is String && v.startsWith('param:')) {
      final name = v.substring(6);
      final pv = ctx.resolveParam(name);
      if (pv is int) return pv;
      _error('SHAPE_PARAM_MISSING', 'Param $name not int', path);
    }
    _error('SHAPE_INVALID_DIM', 'Invalid dim $v', path);
  }

  List<int> _concatAlong(
    Map<String, dynamic> expr,
    ShapeEvalContext ctx,
    String path,
  ) {
    final axis = expr['axis'] ?? 0;
    final inputs = (expr['inputs'] as List).cast<String>();
    if (inputs.isEmpty) {
      _error('SHAPE_RESULT_EMPTY', 'No inputs for concat', path);
    }
    final shapes = <List<int>>[];
    for (final name in inputs) {
      final s =
          ctx.shapeOf(name) ??
          _error('SHAPE_INPUT_UNKNOWN', 'Unknown input $name', path);
      shapes.add(s);
    }
    final rank = shapes.first.length;
    if (shapes.any((s) => s.length != rank)) {
      _error('SHAPE_INCOMPATIBLE', 'Rank mismatch', path);
    }
    if (axis < 0 || axis >= rank) {
      _error('SHAPE_AXIS_OOB', 'Axis $axis out of range', path);
    }
    final out = List<int>.from(shapes.first);
    int sum = 0;
    for (final s in shapes) {
      for (int i = 0; i < rank; i++) {
        if (i == axis) continue;
        if (s[i] != out[i]) {
          _error('SHAPE_INCOMPATIBLE', 'Dim mismatch at $i', path);
        }
      }
      sum += s[axis];
    }
    out[axis] = sum;
    return out;
  }

  List<int> _broadcastPair(
    Map<String, dynamic> expr,
    ShapeEvalContext ctx,
    String path,
  ) {
    final aName = expr['a'];
    final bName = expr['b'];
    final a =
        ctx.shapeOf(aName) ??
        _error('SHAPE_INPUT_UNKNOWN', 'Unknown input $aName', path);
    final b =
        ctx.shapeOf(bName) ??
        _error('SHAPE_INPUT_UNKNOWN', 'Unknown input $bName', path);
    final ar = a.length;
    final br = b.length;
    final r = <int>[];
    final maxR = ar > br ? ar : br;
    for (int i = 0; i < maxR; i++) {
      final da = (i < ar) ? a[ar - 1 - i] : 1;
      final db = (i < br) ? b[br - 1 - i] : 1;
      if (da != db && da != 1 && db != 1) {
        _error('SHAPE_INCOMPATIBLE', 'Cannot broadcast $a and $b', path);
      }
      r.insert(0, da == 1 ? db : da);
    }
    return r;
  }

  void _validateDims(List<int> dims, String path) {
    for (final d in dims) {
      if (d <= 0) {
        _error('SHAPE_INVALID_DIM', 'Non-positive dim $d', path);
      }
    }
  }

  Never _error(String code, String msg, String path) {
    throw ShapeEvalError(code, msg, path);
  }
}

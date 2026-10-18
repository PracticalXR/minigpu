// lib/src/operation/expression.dart
//
// A tiny, dependency-free math-expression engine for per-element CPU stages.
//
// This is the web+native replacement for the removed dart_eval-based dynamic
// CPU stage: instead of interpreting Dart source per frame, an expression is
// parsed ONCE (at stage build) into a tree of Dart closures and then applied
// in a vectorized loop over the input elements.  Closure-tree evaluation is
// AOT-compiled Dart, so an 8k-sample frame with a ~10-node expression costs
// well under a millisecond — 10-100x faster than a bytecode interpreter for
// per-sample loops, and it compiles for the web unchanged (pure Dart, no
// analyzer, no dart:io, no isolates).
//
// Language (everything is a double; booleans are 0.0 / 1.0):
//   literals        1, 2.5, 1e-3
//   variables       x (current element), i (element index), n (element count)
//   identifiers     stage parameters by name, or other INPUT arrays
//                   (indexed at i) — resolved at run time
//   arithmetic      + - * / %    unary -    parentheses
//   comparison      < <= > >= == !=          (yield 1.0 / 0.0)
//   logic           && || !                  (non-zero is true)
//   conditional     cond ? a : b
//   functions       sin cos tan asin acos atan sqrt abs exp log floor ceil
//                   round sign fract (1 arg) · atan2 pow min max (2) ·
//                   clamp(v, lo, hi)
//   constants       pi, e, tau
//
// Example — a Hann window with a 1-sample guard:
//   n > 1 ? x * (0.5 - 0.5 * cos(2 * pi * i / (n - 1))) : x

import 'dart:math' as math;
import 'dart:typed_data';

/// Per-element evaluation scope.  One instance is reused across the whole
/// vectorized loop; only [x] and [i] change per element.
class ExprScope {
  double x = 0.0;
  double i = 0.0;
  double n = 0.0;

  /// Stage parameters bound for this run (name -> value).
  Map<String, double> params = const {};

  /// Sibling input arrays (name -> data), read at the current index.
  Map<String, Float32List> arrays = const {};
}

typedef _Node = double Function(ExprScope s);

/// An expression compiled to a closure tree.
class CompiledExpression {
  CompiledExpression._(this._root, this.source, this.identifiers);

  final _Node _root;

  /// The original expression text (for serialization / diagnostics).
  final String source;

  /// Free identifiers referenced (excluding x/i/n, functions and constants) —
  /// each must resolve to a parameter or an input array at run time.
  final Set<String> identifiers;

  /// Parses [source]; throws [FormatException] with position info on error.
  static CompiledExpression compile(String source) =>
      _Parser(source).parseExpression();

  double evalAt(ExprScope s) => _root(s);

  /// Evaluates the expression over every element of [input], writing into
  /// [out] (same length).  [scope] must have n/params/arrays pre-bound.
  void evalVectorized(Float32List input, Float32List out, ExprScope scope) {
    final root = _root;
    for (var idx = 0; idx < input.length; idx++) {
      scope.x = input[idx];
      scope.i = idx.toDouble();
      out[idx] = root(scope);
    }
  }
}

// ── Tokenizer ──────────────────────────────────────────────────────────────

enum _T { num, ident, op, lparen, rparen, comma, end }

class _Tok {
  _Tok(this.type, this.text, this.pos, [this.value = 0.0]);
  final _T type;
  final String text;
  final int pos;
  final double value;
}

class _Parser {
  _Parser(this.src) {
    _tokenize();
  }

  final String src;
  final List<_Tok> _toks = [];
  int _p = 0;
  final Set<String> _freeIdents = {};

  static const _twoCharOps = ['<=', '>=', '==', '!=', '&&', '||'];
  static const _oneCharOps = ['+', '-', '*', '/', '%', '<', '>', '!', '?', ':'];

  static const Map<String, double> _constants = {
    'pi': math.pi,
    'e': math.e,
    'tau': math.pi * 2,
  };

  void _fail(int pos, String message) {
    throw FormatException('Expression error at $pos: $message', src, pos);
  }

  void _tokenize() {
    var i = 0;
    while (i < src.length) {
      final c = src[i];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        i++;
        continue;
      }
      if (c == '(') {
        _toks.add(_Tok(_T.lparen, c, i));
        i++;
        continue;
      }
      if (c == ')') {
        _toks.add(_Tok(_T.rparen, c, i));
        i++;
        continue;
      }
      if (c == ',') {
        _toks.add(_Tok(_T.comma, c, i));
        i++;
        continue;
      }
      if (i + 1 < src.length && _twoCharOps.contains(src.substring(i, i + 2))) {
        _toks.add(_Tok(_T.op, src.substring(i, i + 2), i));
        i += 2;
        continue;
      }
      if (_oneCharOps.contains(c)) {
        _toks.add(_Tok(_T.op, c, i));
        i++;
        continue;
      }
      if (_isDigit(c) ||
          (c == '.' && i + 1 < src.length && _isDigit(src[i + 1]))) {
        final start = i;
        while (i < src.length && _isDigit(src[i])) {
          i++;
        }
        if (i < src.length && src[i] == '.') {
          i++;
          while (i < src.length && _isDigit(src[i])) {
            i++;
          }
        }
        if (i < src.length && (src[i] == 'e' || src[i] == 'E')) {
          i++;
          if (i < src.length && (src[i] == '+' || src[i] == '-')) i++;
          while (i < src.length && _isDigit(src[i])) {
            i++;
          }
        }
        final text = src.substring(start, i);
        final value = double.tryParse(text);
        if (value == null) _fail(start, "bad number '$text'");
        _toks.add(_Tok(_T.num, text, start, value!));
        continue;
      }
      if (_isIdentStart(c)) {
        final start = i;
        while (i < src.length && _isIdentChar(src[i])) {
          i++;
        }
        _toks.add(_Tok(_T.ident, src.substring(start, i), start));
        continue;
      }
      _fail(i, "unexpected character '$c'");
    }
    _toks.add(_Tok(_T.end, '', src.length));
  }

  static bool _isDigit(String c) =>
      c.codeUnitAt(0) >= 0x30 && c.codeUnitAt(0) <= 0x39;
  static bool _isIdentStart(String c) {
    final u = c.codeUnitAt(0);
    return (u >= 0x61 && u <= 0x7a) || (u >= 0x41 && u <= 0x5a) || u == 0x5f;
  }

  static bool _isIdentChar(String c) => _isIdentStart(c) || _isDigit(c);

  _Tok get _cur => _toks[_p];

  bool _matchOp(String op) {
    if (_cur.type == _T.op && _cur.text == op) {
      _p++;
      return true;
    }
    return false;
  }

  // ── Grammar (precedence climbing) ────────────────────────────────────────
  //   ternary > or > and > equality > comparison > additive > multiplicative
  //   > unary > primary/call

  CompiledExpression parseExpression() {
    final root = _ternary();
    if (_cur.type != _T.end) {
      _fail(_cur.pos, "unexpected '${_cur.text}'");
    }
    return CompiledExpression._(root.node, src, _freeIdents);
  }

  _Expr _ternary() {
    final cond = _or();
    if (!_matchOp('?')) return cond;
    final thenE = _ternary();
    if (!_matchOp(':')) _fail(_cur.pos, "expected ':' in conditional");
    final elseE = _ternary();
    if (cond.isConst) {
      return cond.value != 0.0 ? thenE : elseE;
    }
    final c = cond.node, t = thenE.node, f = elseE.node;
    return _Expr((s) => c(s) != 0.0 ? t(s) : f(s));
  }

  _Expr _or() {
    var left = _and();
    while (_matchOp('||')) {
      final right = _and();
      left = _fold2(
        left,
        right,
        (a, b) => (a != 0.0 || b != 0.0) ? 1.0 : 0.0,
        shortCircuit: (l, r) =>
            (s) => l(s) != 0.0 ? 1.0 : (r(s) != 0.0 ? 1.0 : 0.0),
      );
    }
    return left;
  }

  _Expr _and() {
    var left = _equality();
    while (_matchOp('&&')) {
      final right = _equality();
      left = _fold2(
        left,
        right,
        (a, b) => (a != 0.0 && b != 0.0) ? 1.0 : 0.0,
        shortCircuit: (l, r) =>
            (s) => l(s) == 0.0 ? 0.0 : (r(s) != 0.0 ? 1.0 : 0.0),
      );
    }
    return left;
  }

  _Expr _equality() {
    var left = _comparison();
    while (true) {
      if (_matchOp('==')) {
        left = _fold2(left, _comparison(), (a, b) => a == b ? 1.0 : 0.0);
      } else if (_matchOp('!=')) {
        left = _fold2(left, _comparison(), (a, b) => a != b ? 1.0 : 0.0);
      } else {
        return left;
      }
    }
  }

  _Expr _comparison() {
    var left = _additive();
    while (true) {
      if (_matchOp('<=')) {
        left = _fold2(left, _additive(), (a, b) => a <= b ? 1.0 : 0.0);
      } else if (_matchOp('>=')) {
        left = _fold2(left, _additive(), (a, b) => a >= b ? 1.0 : 0.0);
      } else if (_matchOp('<')) {
        left = _fold2(left, _additive(), (a, b) => a < b ? 1.0 : 0.0);
      } else if (_matchOp('>')) {
        left = _fold2(left, _additive(), (a, b) => a > b ? 1.0 : 0.0);
      } else {
        return left;
      }
    }
  }

  _Expr _additive() {
    var left = _multiplicative();
    while (true) {
      if (_matchOp('+')) {
        left = _fold2(left, _multiplicative(), (a, b) => a + b);
      } else if (_matchOp('-')) {
        left = _fold2(left, _multiplicative(), (a, b) => a - b);
      } else {
        return left;
      }
    }
  }

  _Expr _multiplicative() {
    var left = _unary();
    while (true) {
      if (_matchOp('*')) {
        left = _fold2(left, _unary(), (a, b) => a * b);
      } else if (_matchOp('/')) {
        left = _fold2(left, _unary(), (a, b) => a / b);
      } else if (_matchOp('%')) {
        left = _fold2(left, _unary(), (a, b) => a % b);
      } else {
        return left;
      }
    }
  }

  _Expr _unary() {
    if (_matchOp('-')) {
      final e = _unary();
      if (e.isConst) return _Expr.constant(-e.value);
      final n = e.node;
      return _Expr((s) => -n(s));
    }
    if (_matchOp('!')) {
      final e = _unary();
      if (e.isConst) return _Expr.constant(e.value == 0.0 ? 1.0 : 0.0);
      final n = e.node;
      return _Expr((s) => n(s) == 0.0 ? 1.0 : 0.0);
    }
    return _primary();
  }

  _Expr _primary() {
    final t = _cur;
    switch (t.type) {
      case _T.num:
        _p++;
        return _Expr.constant(t.value);
      case _T.lparen:
        _p++;
        final inner = _ternary();
        if (_cur.type != _T.rparen) _fail(_cur.pos, "expected ')'");
        _p++;
        return inner;
      case _T.ident:
        _p++;
        if (_cur.type == _T.lparen) {
          return _call(t);
        }
        return _identifier(t);
      default:
        _fail(t.pos, "unexpected '${t.text}'");
    }
    throw StateError('unreachable');
  }

  _Expr _identifier(_Tok t) {
    final name = t.text;
    final constant = _constants[name];
    if (constant != null) return _Expr.constant(constant);
    switch (name) {
      case 'x':
        return _Expr((s) => s.x);
      case 'i':
        return _Expr((s) => s.i);
      case 'n':
        return _Expr((s) => s.n);
    }
    _freeIdents.add(name);
    return _Expr((s) {
      final p = s.params[name];
      if (p != null) return p;
      final a = s.arrays[name];
      if (a != null) {
        final idx = s.i.toInt();
        return (idx >= 0 && idx < a.length) ? a[idx] : 0.0;
      }
      throw StateError(
        "Expression identifier '$name' is neither a parameter nor an input",
      );
    });
  }

  _Expr _call(_Tok nameTok) {
    // _cur is the lparen.
    _p++;
    final args = <_Expr>[];
    if (_cur.type != _T.rparen) {
      args.add(_ternary());
      while (_cur.type == _T.comma) {
        _p++;
        args.add(_ternary());
      }
    }
    if (_cur.type != _T.rparen) _fail(_cur.pos, "expected ')' after arguments");
    _p++;

    final name = nameTok.text;
    _Expr expect(int arity, _Expr Function() build) {
      if (args.length != arity) {
        _fail(
          nameTok.pos,
          '$name() takes $arity argument(s), got ${args.length}',
        );
      }
      return build();
    }

    _Expr f1(double Function(double) fn) => expect(1, () {
      final a = args[0];
      if (a.isConst) return _Expr.constant(fn(a.value));
      final an = a.node;
      return _Expr((s) => fn(an(s)));
    });

    _Expr f2(double Function(double, double) fn) => expect(2, () {
      final a = args[0], b = args[1];
      if (a.isConst && b.isConst) return _Expr.constant(fn(a.value, b.value));
      final an = a.node, bn = b.node;
      return _Expr((s) => fn(an(s), bn(s)));
    });

    switch (name) {
      case 'sin':
        return f1(math.sin);
      case 'cos':
        return f1(math.cos);
      case 'tan':
        return f1(math.tan);
      case 'asin':
        return f1(math.asin);
      case 'acos':
        return f1(math.acos);
      case 'atan':
        return f1(math.atan);
      case 'sqrt':
        return f1(math.sqrt);
      case 'abs':
        return f1((v) => v.abs());
      case 'exp':
        return f1(math.exp);
      case 'log':
        return f1(math.log);
      case 'floor':
        return f1((v) => v.floorToDouble());
      case 'ceil':
        return f1((v) => v.ceilToDouble());
      case 'round':
        return f1((v) => v.roundToDouble());
      case 'sign':
        return f1((v) => v.sign);
      case 'fract':
        return f1((v) => v - v.floorToDouble());
      case 'atan2':
        return f2(math.atan2);
      case 'pow':
        return f2((a, b) => math.pow(a, b).toDouble());
      case 'min':
        return f2(math.min);
      case 'max':
        return f2(math.max);
      case 'clamp':
        return expect(3, () {
          final v = args[0].node, lo = args[1].node, hi = args[2].node;
          return _Expr((s) => v(s).clamp(lo(s), hi(s)).toDouble());
        });
      default:
        _fail(nameTok.pos, "unknown function '$name'");
    }
    throw StateError('unreachable');
  }

  /// Binary node with constant folding.
  _Expr _fold2(
    _Expr a,
    _Expr b,
    double Function(double, double) fn, {
    _Node Function(_Node, _Node)? shortCircuit,
  }) {
    if (a.isConst && b.isConst) return _Expr.constant(fn(a.value, b.value));
    if (shortCircuit != null) return _Expr(shortCircuit(a.node, b.node));
    final an = a.node, bn = b.node;
    return _Expr((s) => fn(an(s), bn(s)));
  }
}

/// A parse result: a closure plus a constant marker for folding.
class _Expr {
  _Expr(this.node) : isConst = false, value = 0.0;
  _Expr.constant(double v) : value = v, isConst = true, node = _constNode(v);

  static _Node _constNode(double v) =>
      (s) => v;

  final _Node node;
  final bool isConst;
  final double value;
}

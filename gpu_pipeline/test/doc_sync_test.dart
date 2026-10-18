import 'dart:io';

import 'package:test/test.dart';

// Every code block in doc/ must be an excerpt of doc_examples_test.dart, so
// the examples the guides show are the ones that run.
void main() {
  final source = File('test/doc_examples_test.dart').readAsLinesSync();
  final guides = Directory(
    'doc',
  ).listSync().whereType<File>().where((f) => f.path.endsWith('.md'));

  for (final guide in guides) {
    test('${guide.uri.pathSegments.last} quotes only tested code', () {
      final lines = guide.readAsLinesSync();
      for (var i = 0; i < lines.length; i++) {
        if (!lines[i].startsWith('```')) continue;
        final end = lines.indexWhere((l) => l.startsWith('```'), i + 1);
        expect(end, isNot(-1), reason: '${guide.path}:${i + 1} is not closed');
        expect(
          _contains(source, lines.sublist(i + 1, end)),
          isTrue,
          reason: '${guide.path}:${i + 1} is not in doc_examples_test.dart',
        );
        i = end;
      }
    });
  }
}

/// Whether [block] appears in [source] as consecutive lines, all indented by
/// the same amount.
bool _contains(List<String> source, List<String> block) {
  for (var s = 0; s + block.length <= source.length; s++) {
    final first = source[s];
    if (!first.endsWith(block.first)) continue;
    final indent = first.substring(0, first.length - block.first.length);
    if (indent.trim().isNotEmpty) continue;
    var found = true;
    for (var j = 0; j < block.length && found; j++) {
      final line = source[s + j];
      found = block[j].isEmpty
          ? line.trim().isEmpty
          : line == indent + block[j];
    }
    if (found) return true;
  }
  return false;
}

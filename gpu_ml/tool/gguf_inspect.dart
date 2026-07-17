// Dumps GGUF header/metadata/tensor-directory info without loading tensor
// data: reads only the first chunk of the file (header + KV + directory).
//
//   dart run tool/gguf_inspect.dart <path.gguf> [--tensors]
//
// For quant types the library doesn't know yet, derives bytes-per-element
// from consecutive tensor offsets so new block layouts can be reverse-sized.
import 'dart:io';
import 'dart:typed_data';

import 'package:gpu_ml/gpu_ml.dart';

Future<void> main(List<String> args) async {
  if (args.isEmpty) {
    print('usage: dart run tool/gguf_inspect.dart <path.gguf> [--tensors]');
    exit(1);
  }
  final path = args[0];
  final showTensors = args.contains('--tensors');

  final file = File(path);
  final fileLen = await file.length();
  // Header + metadata (incl. tokenizer vocab arrays) + directory; 256MB cap.
  final headerLen = fileLen < 256 * 1024 * 1024 ? fileLen : 256 * 1024 * 1024;
  final raf = await file.open();
  final headBytes = Uint8List(headerLen);
  await raf.readInto(headBytes);
  await raf.close();

  final f = GgufFile.parse(headBytes);
  print('file: $path');
  print('size: ${(fileLen / (1024 * 1024 * 1024)).toStringAsFixed(2)} GB');
  print('gguf version: ${f.version}');
  print('alignment: ${f.alignment}   dataOffset: ${f.dataOffset}');
  print('tensor count: ${f.tensors.length}');
  print('');

  print('--- metadata (${f.metadata.length} keys) ---');
  final keys = f.metadata.keys.toList()..sort();
  for (final k in keys) {
    final v = f.metadata[k];
    if (v is List) {
      final preview = v.take(6).join(', ');
      print('$k: List[${v.length}] [$preview${v.length > 6 ? ', ...' : ''}]');
    } else if (v is String && v.length > 120) {
      print('$k: "${v.substring(0, 120)}..." (${v.length} chars)');
    } else {
      print('$k: $v');
    }
  }
  print('');

  // Type histogram + derived bytes-per-element for unknown types.
  final byType = <int, List<GgufTensorInfo>>{};
  for (final t in f.tensors) {
    byType.putIfAbsent(t.type, () => []).add(t);
  }
  final sorted = f.tensors.toList()..sort((a, b) => a.offset - b.offset);

  print('--- tensor types ---');
  final typeIds = byType.keys.toList()..sort();
  for (final ty in typeIds) {
    final ts = byType[ty]!;
    final known = ggmlTypeTraits.containsKey(ty);
    // Derive average bytes/element from offset deltas of consecutive tensors
    // of this type (alignment padding <= 31B noise).
    double? bpe;
    for (int i = 0; i < sorted.length - 1; i++) {
      if (sorted[i].type == ty) {
        final delta = sorted[i + 1].offset - sorted[i].offset;
        bpe = delta / sorted[i].elementCount;
        break;
      }
    }
    print(
      'type $ty: ${ts.length} tensors'
      '${known ? ' (known)' : ' (UNKNOWN to gpu_tensor)'}'
      '${bpe != null ? '  ~${bpe.toStringAsFixed(4)} bytes/element' : ''}'
      '  e.g. ${ts.first.name} ne=${ts.first.ne}',
    );
  }

  if (showTensors) {
    print('');
    print('--- tensors ---');
    for (final t in sorted) {
      print('${t.name}  ne=${t.ne}  type=${t.type}  offset=${t.offset}');
    }
  }
}

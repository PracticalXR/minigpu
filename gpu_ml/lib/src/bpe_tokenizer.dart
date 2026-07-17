import 'dart:convert';

/// GPT-2-style byte-level BPE tokenizer built from GGUF metadata
/// (`tokenizer.ggml.tokens` / `.merges` / `.token_type`), with the
/// llama.cpp `qwen35` pre-tokenizer regex.  Web-safe (pure Dart).
///
/// v1 scope: plain-text encode/decode for generation smoke tests.  Special
/// tokens (chat template markers) are decoded but never produced by encode;
/// prompt-side special-token splitting comes with the chat template work.
class BpeTokenizer {
  BpeTokenizer._({
    required this.tokens,
    required Map<String, int> tokenToId,
    required Map<String, int> mergeRanks,
    required this.bosId,
    required this.eosId,
    required this.tokenTypes,
  })  : _tokenToId = tokenToId,
        _mergeRanks = mergeRanks;

  final List<String> tokens;
  final List<int> tokenTypes; // ggml token types; 3 = control/special
  final Map<String, int> _tokenToId;
  final Map<String, int> _mergeRanks;
  final int bosId;
  final int eosId;

  static const int typeControl = 3;

  /// llama.cpp LLAMA_VOCAB_PRE_TYPE_QWEN35 regex.
  static final RegExp _pre = RegExp(
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+",
    unicode: true,
  );

  // GPT-2 byte <-> unicode: printable bytes map to themselves; the rest map
  // to U+0100+n in order.
  static final List<String> _byteToChar = _buildByteToChar();
  static final Map<String, int> _charToByte = {
    for (int b = 0; b < 256; b++) _byteToChar[b]: b,
  };

  static List<String> _buildByteToChar() {
    final printable = <int>[
      for (int b = 0x21; b <= 0x7E; b++) b,
      for (int b = 0xA1; b <= 0xAC; b++) b,
      for (int b = 0xAE; b <= 0xFF; b++) b,
    ];
    final out = List<String>.filled(256, '');
    int n = 0;
    for (int b = 0; b < 256; b++) {
      if (printable.contains(b)) {
        out[b] = String.fromCharCode(b);
      } else {
        out[b] = String.fromCharCode(0x100 + n);
        n++;
      }
    }
    return out;
  }

  static BpeTokenizer fromGgufMetadata(Map<String, dynamic> metadata) {
    final tokens = (metadata['tokenizer.ggml.tokens'] as List).cast<String>();
    final mergesList =
        (metadata['tokenizer.ggml.merges'] as List).cast<String>();
    final types = (metadata['tokenizer.ggml.token_type'] as List?)
            ?.cast<int>() ??
        List<int>.filled(tokens.length, 1);
    final tokenToId = <String, int>{
      for (int i = 0; i < tokens.length; i++) tokens[i]: i,
    };
    final mergeRanks = <String, int>{
      for (int i = 0; i < mergesList.length; i++) mergesList[i]: i,
    };
    return BpeTokenizer._(
      tokens: tokens,
      tokenToId: tokenToId,
      mergeRanks: mergeRanks,
      bosId: (metadata['tokenizer.ggml.bos_token_id'] as int?) ?? -1,
      eosId: (metadata['tokenizer.ggml.eos_token_id'] as int?) ?? -1,
      tokenTypes: types,
    );
  }

  /// Encodes plain text (no special-token parsing, no BOS prepend — qwen
  /// does not use BOS for plain text).
  List<int> encode(String text) {
    final ids = <int>[];
    for (final m in _pre.allMatches(text)) {
      final piece = m.group(0)!;
      // Bytes -> byte-level chars.
      final chars = [
        for (final b in utf8.encode(piece)) _byteToChar[b],
      ];
      // Greedy lowest-rank merges.
      var parts = chars;
      while (parts.length > 1) {
        int bestRank = 0x7FFFFFFF;
        int bestIdx = -1;
        for (int i = 0; i < parts.length - 1; i++) {
          final rank = _mergeRanks['${parts[i]} ${parts[i + 1]}'];
          if (rank != null && rank < bestRank) {
            bestRank = rank;
            bestIdx = i;
          }
        }
        if (bestIdx < 0) break;
        parts = [
          ...parts.sublist(0, bestIdx),
          parts[bestIdx] + parts[bestIdx + 1],
          ...parts.sublist(bestIdx + 2),
        ];
      }
      for (final p in parts) {
        final id = _tokenToId[p];
        if (id != null) {
          ids.add(id);
        } else {
          // Fall back to per-byte tokens.
          for (final ch in p.split('')) {
            final byteId = _tokenToId[ch];
            if (byteId != null) ids.add(byteId);
          }
        }
      }
    }
    return ids;
  }

  /// Encodes a token that must exist verbatim in the vocab (chat-template
  /// markers like `<|im_start|>`).
  int special(String token) {
    final id = _tokenToId[token];
    if (id == null) {
      throw Exception("special token '$token' not in vocab");
    }
    return id;
  }

  bool isControl(int id) =>
      id >= 0 && id < tokenTypes.length && tokenTypes[id] == typeControl;

  /// Decodes token ids to text.  Control tokens are skipped unless
  /// [keepSpecial].
  String decode(List<int> ids, {bool keepSpecial = false}) {
    final bytes = <int>[];
    final sb = StringBuffer();
    void flushBytes() {
      if (bytes.isEmpty) return;
      sb.write(utf8.decode(bytes, allowMalformed: true));
      bytes.clear();
    }

    for (final id in ids) {
      if (id < 0 || id >= tokens.length) continue;
      if (isControl(id)) {
        if (keepSpecial) {
          flushBytes();
          sb.write(tokens[id]);
        }
        continue;
      }
      for (final ch in tokens[id].split('')) {
        final b = _charToByte[ch];
        if (b != null) {
          bytes.add(b);
        } else {
          flushBytes();
          sb.write(ch);
        }
      }
    }
    flushBytes();
    return sb.toString();
  }
}

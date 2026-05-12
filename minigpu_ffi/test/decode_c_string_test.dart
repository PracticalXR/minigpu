// Tests for the _decodeCString helper used in MinigpuFfi.setLogCallback.
//
// The helper reads a null-terminated C string via FFI and decodes it with
// Utf8Decoder(allowMalformed: true) so that non-UTF-8 bytes from Dawn/driver
// log messages never cause a FormatException.  These tests verify both the
// happy path and the malformed-byte cases that motivated the fix.

import 'dart:convert';
import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:test/test.dart';

// ---------------------------------------------------------------------------
// Mirror of the private _decodeCString implementation in minigpu_ffi.dart.
// Any change to the production helper must be reflected here.
// ---------------------------------------------------------------------------
String decodeCString(Pointer<Char> ptr) {
  if (ptr.address == 0) return '';
  final bytes = ptr.cast<Uint8>();
  var len = 0;
  while (bytes[len] != 0) len++;
  return const Utf8Decoder(
    allowMalformed: true,
  ).convert(Uint8List.view(bytes.asTypedList(len).buffer, 0, len));
}

// Allocates a native buffer containing [bytes] followed by a null terminator.
// Caller is responsible for freeing with [malloc.free].
Pointer<Char> _alloc(List<int> bytes) {
  final ptr = malloc<Uint8>(bytes.length + 1);
  for (var i = 0; i < bytes.length; i++) {
    ptr[i] = bytes[i] & 0xFF;
  }
  ptr[bytes.length] = 0;
  return ptr.cast<Char>();
}

void main() {
  group('decodeCString', () {
    // ------------------------------------------------------------------
    // Null / empty
    // ------------------------------------------------------------------
    test('null pointer (address 0) returns empty string', () {
      expect(decodeCString(Pointer.fromAddress(0)), equals(''));
    });

    test('empty C string (immediate null terminator) returns empty string', () {
      final ptr = _alloc([]);
      addTearDown(() => malloc.free(ptr));
      expect(decodeCString(ptr), equals(''));
    });

    // ------------------------------------------------------------------
    // Valid encodings
    // ------------------------------------------------------------------
    test('pure ASCII is decoded correctly', () {
      const input = 'Hello, World!';
      final ptr = _alloc(input.codeUnits);
      addTearDown(() => malloc.free(ptr));
      expect(decodeCString(ptr), equals(input));
    });

    test('valid UTF-8 multi-byte characters are decoded correctly', () {
      // 'café' → [0x63, 0x61, 0x66, 0xC3, 0xA9]
      const input = 'café';
      final ptr = _alloc(utf8.encode(input));
      addTearDown(() => malloc.free(ptr));
      expect(decodeCString(ptr), equals(input));
    });

    test('Unicode BMP character (euro sign U+20AC) is decoded correctly', () {
      // '€' → [0xE2, 0x82, 0xAC]
      const input = '€';
      final ptr = _alloc(utf8.encode(input));
      addTearDown(() => malloc.free(ptr));
      expect(decodeCString(ptr), equals(input));
    });

    // ------------------------------------------------------------------
    // Malformed / non-UTF-8 bytes — the core of the fix
    // ------------------------------------------------------------------
    test('isolated 0xFF byte does not throw (was the original crash)', () {
      // Before the fix, msg.cast<Utf8>().toDartString() would throw:
      //   FormatException: Unexpected extension byte (at offset 0)
      final ptr = _alloc([0xFF]);
      addTearDown(() => malloc.free(ptr));
      expect(() => decodeCString(ptr), returnsNormally);
    });

    test('isolated continuation byte 0x80 does not throw', () {
      final ptr = _alloc([0x80]);
      addTearDown(() => malloc.free(ptr));
      expect(() => decodeCString(ptr), returnsNormally);
    });

    test('Latin-1 high bytes (0xFF 0xFE) do not throw', () {
      final ptr = _alloc([0xFF, 0xFE, 0x41]); // invalid, invalid, 'A'
      addTearDown(() => malloc.free(ptr));
      expect(() => decodeCString(ptr), returnsNormally);
      // The valid ASCII byte 'A' must survive regardless of replacement policy.
      expect(decodeCString(ptr), endsWith('A'));
    });

    test('mixed valid UTF-8 and invalid bytes does not throw', () {
      // 'OK' + 0xFF (bad byte) + 'Z'
      final bytes = [...utf8.encode('OK'), 0xFF, 0x5A]; // 0x5A = 'Z'
      final ptr = _alloc(bytes);
      addTearDown(() => malloc.free(ptr));
      expect(() => decodeCString(ptr), returnsNormally);
      final result = decodeCString(ptr);
      expect(result, startsWith('OK'));
      expect(result, endsWith('Z'));
    });

    test('multi-byte sequence truncated at null terminator does not throw', () {
      // Start of a 3-byte sequence but only one byte before the null — invalid.
      final ptr = _alloc([0xE2]); // first byte of '€' with no continuation
      addTearDown(() => malloc.free(ptr));
      expect(() => decodeCString(ptr), returnsNormally);
    });

    // ------------------------------------------------------------------
    // Documents why allowMalformed: false is insufficient
    // ------------------------------------------------------------------
    test(
      'Utf8Decoder(allowMalformed: false) throws on 0xFF — motivates fix',
      () {
        expect(
          () => const Utf8Decoder(
            allowMalformed: false,
          ).convert(Uint8List.fromList([0xFF])),
          throwsFormatException,
        );
      },
    );

    // ------------------------------------------------------------------
    // Null-terminator position
    // ------------------------------------------------------------------
    test('null terminator stops reading at the correct position', () {
      // Native layout: 'A', 'B', '\0', 'C', 'D'
      // Only 'AB' should be returned.
      final ptr = malloc<Uint8>(5);
      addTearDown(() => malloc.free(ptr));
      ptr[0] = 0x41; // 'A'
      ptr[1] = 0x42; // 'B'
      ptr[2] = 0x00; // null terminator
      ptr[3] = 0x43; // 'C' — beyond null, must NOT appear
      ptr[4] = 0x44; // 'D' — beyond null, must NOT appear
      expect(decodeCString(ptr.cast<Char>()), equals('AB'));
    });

    test(
      'realistic Dawn log line with trailing newline is decoded correctly',
      () {
        const line = '[Dawn] D3D12 backend initialized.\n';
        final ptr = _alloc(utf8.encode(line));
        addTearDown(() => malloc.free(ptr));
        expect(decodeCString(ptr), equals(line));
      },
    );
  });

  // ---------------------------------------------------------------------------
  // String lifetime — regression tests for the dangling-pointer bug.
  //
  // Root cause: NativeCallable.listener posts arguments to the Dart event loop
  // asynchronously.  The Pointer<Char> was copied as a raw integer (address);
  // by the time Dart ran the closure the C++ stack buffer (char buffer[1024])
  // had been deallocated, yielding partial strings like "[Minigpu] ssed".
  //
  // Fix: C++ heap-allocates (malloc+memcpy) before calling the callback;
  // Dart reads the string then calls mgpuFreeLogMessage().
  // ---------------------------------------------------------------------------
  group('log callback string lifetime', () {
    test('decoded Dart string is independent of native buffer — '
        'overwriting native bytes does not corrupt the result '
        '(regression: "[Minigpu] ssed" partial-overwrite scenario)', () {
      // Observed bug: only the last 4 characters of "processed" arrived
      // because the C++ stack frame was partially reused before Dart read it.
      const full = '[Minigpu] Buffer released and events processed';
      final ptr = _alloc(utf8.encode(full));
      final dartStr = decodeCString(ptr);

      // Overwrite the native buffer — simulates the C++ stack frame being
      // reused after the function returns (partial overwrite of log text).
      final raw = ptr.cast<Uint8>();
      for (var i = 0; i < full.length; i++) {
        raw[i] = 0x73; // 's' — what the partial overwrite left behind
      }
      malloc.free(ptr);

      // The Dart string must be the FULL original — not '[Minigpu] ssed'.
      expect(dartStr, equals(full));
    });

    test('heap-copy pattern: allocate → decode → free preserves full message '
        '(simulates mgpuFreeLogMessage called after decodeCString)', () {
      const msg = '[Dawn] Adapter: NVIDIA GeForce RTX 4080 (Vulkan) — selected';
      // Simulates what the fixed C++ logger does: heap-allocate a copy of
      // the formatted string and hand the pointer to the Dart callback.
      final heapPtr = _alloc(utf8.encode(msg));

      // Listener body: read string then free (mgpuFreeLogMessage).
      final received = decodeCString(heapPtr);
      malloc.free(heapPtr); // pointer now invalid

      // The Dart string must be intact after the native memory is freed.
      expect(received, equals(msg));
    });

    test('message exactly 1024 bytes decoded completely — '
        'no truncation at former stack-buffer boundary', () {
      // The old C++ logger used char buffer[1024].  A log line that fills
      // the buffer should arrive with every character intact.
      // Total length with null terminator = 1024 bytes → content = 1023.
      final content = 'x' * 1023;
      final ptr = _alloc(utf8.encode(content));
      addTearDown(() => malloc.free(ptr));
      final result = decodeCString(ptr);
      expect(result.length, equals(1023));
      expect(result, equals(content));
    });

    test('message with printf format specifiers is decoded verbatim — '
        'not re-interpreted through snprintf', () {
      // If the message were accidentally passed through printf again, %s/%d
      // would expand or crash.  Verify the raw text is preserved.
      const msg = '[Minigpu] device %s vendor %d format %%';
      final ptr = _alloc(utf8.encode(msg));
      addTearDown(() => malloc.free(ptr));
      expect(decodeCString(ptr), equals(msg));
    });

    test('async event-loop delay: Dart string decoded before free is intact '
        'when accessed after Future resolves', () async {
      // Simulates the listener body: decode synchronously, free the native
      // buffer, then yield to the event loop. The Dart string must be
      // accessible after the await — it is a value, not a pointer.
      const msg = '[Minigpu] Buffer released and events processed';
      final ptr = _alloc(utf8.encode(msg));

      String? received;
      await Future.microtask(() {
        received = decodeCString(ptr);
        malloc.free(ptr); // simulate mgpuFreeLogMessage
      });

      expect(received, equals(msg));
    });
  });
}

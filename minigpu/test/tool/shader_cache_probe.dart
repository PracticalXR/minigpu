/// Standalone shader-cache probe.
///
/// Runs one minigpu context, compiles [--kernels] distinct compute kernels,
/// dispatches each once, and prints a single JSON line of cache counters.
///
/// It exists because the interesting question — "does a SECOND PROCESS get
/// cache hits?" — cannot be answered from inside one test process: the native
/// context is process-global and the cache's in-memory state would be shared.
/// `shader_cache_test.dart` spawns this twice against one temp directory.
///
/// Usage:
///   dart test/tool/shader_cache_probe.dart --dir=<path> [--kernels=3]
///                                          [--extra-key=<s>] [--disable]
///                                          [--cap=<bytes>] [--verify]
///
/// Output (stdout, last line): {"hits":N,"misses":N,...,"checksum":N}
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:minigpu/minigpu.dart';

String? _arg(List<String> args, String name) {
  final prefix = '--$name=';
  for (final a in args) {
    if (a.startsWith(prefix)) return a.substring(prefix.length);
  }
  return null;
}

bool _flag(List<String> args, String name) => args.contains('--$name');

/// A distinct kernel per [index]. The body is deliberately a few unrolled
/// passes rather than one multiply: a trivial kernel compiles too fast for a
/// cold/warm comparison to mean anything, and the point of the cache is
/// kernels whose compile time is measurable.
String _kernel(int index) {
  final buf = StringBuffer('''
@group(0) @binding(0) var<storage, read_write> input_0: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_0: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&input_0)) { return; }
  var acc = input_0[i];
''');
  // The literal constants differ per kernel, so each is a distinct cache key.
  for (var pass = 0; pass < 12; pass++) {
    final k = (index + 1) * 100 + pass;
    buf.writeln('  acc = acc * ${k / 97.0} + sin(acc * ${k / 31.0});');
    buf.writeln('  acc = acc - floor(acc / ${k + 3}.0) * ${k + 3}.0;');
  }
  buf.writeln('  output_0[i] = acc;');
  buf.writeln('}');
  return buf.toString();
}

Future<void> main(List<String> args) async {
  final dir = _arg(args, 'dir');
  final kernels = int.tryParse(_arg(args, 'kernels') ?? '3') ?? 3;
  final extraKey = _arg(args, 'extra-key');
  final cap = _arg(args, 'cap');
  final disable = _flag(args, 'disable');
  final verify = _flag(args, 'verify');

  // Pre-init: the isolation key is baked into the device at creation, so all
  // of this has to land before init().
  Minigpu.configureShaderCache(
    enabled: disable ? false : true,
    directory: dir,
    extraKey: extraKey,
    maxBytes: cap == null ? null : int.parse(cap),
  );

  final gpu = Minigpu();
  await gpu.init();

  const n = 256;
  var checksum = 0;
  try {
    for (var k = 0; k < kernels; k++) {
      final input = gpu.createBuffer(n * 4, BufferDataType.float32);
      final output = gpu.createBuffer(n * 4, BufferDataType.float32);
      input.write(
        Float32List.fromList(List.generate(n, (i) => (i + 1).toDouble())),
        n,
        dataType: BufferDataType.float32,
      );

      final shader = gpu.createComputeShader()..loadKernelString(_kernel(k));
      shader.setBuffer('input_0', input);
      shader.setBuffer('output_0', output);
      await shader.dispatch((n + 63) ~/ 64, 1, 1);

      if (verify) {
        // Byte-identical output cold vs warm is the real acceptance test: a
        // shader cache that changes one output value is a bug, and a
        // checksum over the raw bits is what proves it did not.
        final out = Float32List(n);
        await output.read(out, n, dataType: BufferDataType.float32);
        final bytes = Uint8List.view(out.buffer);
        var h = 0x811c9dc5;
        for (final b in bytes) {
          h = ((h ^ b) * 0x01000193) & 0xffffffff;
        }
        checksum = (checksum ^ h) & 0xffffffff;
      }

      shader.destroy();
      input.destroy();
      output.destroy();
    }

    final s = Minigpu.shaderCacheStats;
    stdout.writeln(
      jsonEncode({
        'hits': s?.hits ?? -1,
        'misses': s?.misses ?? -1,
        'stores': s?.stores ?? -1,
        'storeFailures': s?.storeFailures ?? -1,
        'evictions': s?.evictions ?? -1,
        'entryCount': s?.entryCount ?? -1,
        'bytesOnDisk': s?.bytesOnDisk ?? -1,
        'pipelineCreateMs': s?.pipelineCreateMs ?? -1,
        'enabled': s?.enabled ?? false,
        'directory': Minigpu.shaderCacheDirectory ?? '',
        'checksum': checksum,
      }),
    );
  } finally {
    await gpu.destroy();
  }
  // The native worker thread can outlive the last await; exit explicitly so a
  // parent Process.run() is never left waiting on a lingering isolate.
  exit(0);
}

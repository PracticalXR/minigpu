// Standalone dart2wasm acceptance, served with the generated minigpu assets.
// No Flutter shell or substitute GPU device is involved.
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

import 'package:minigpu_web/bindings/minigpu_bindings.dart' as gpu;
import 'package:minigpu_web/bindings/webgpu_interop.dart';

void check(bool condition, String message) {
  if (!condition) throw StateError(message);
}

Future<void> main() async {
  final buffers = <gpu.MGPUBuffer>[];
  gpu.MGPUComputeShader? shader;
  var initialized = false;
  var outcome = 'passed';
  try {
    await gpu.mgpuInitializeContext();
    initialized = true;
    check(
      gpu.hasBufferDescriptorV1 && gpu.hasBufferBindingV1,
      'Generated module lacks the versioned buffer APIs',
    );
    gpu.MGPUBuffer allocate(int bytes, int usage) {
      final buffer = gpu.mgpuCreateBufferWithUsage(bytes, usage);
      check(buffer != null, 'Allocation refused');
      buffers.add(buffer!);
      final actual = getWebGpuJsObject(gpu.mgpuGetWGPUBufferHandle(buffer))!;
      check(
        actual.getProperty<JSNumber>('size'.toJS).toDartInt == bytes,
        'Allocation changed the byte size',
      );
      check(
        actual.getProperty<JSNumber>('usage'.toJS).toDartInt == usage,
        'Allocation changed the usage mask',
      );
      return buffer;
    }

    final uniform = allocate(768, 64 | 8);
    final source = allocate(768, 128 | 8);
    final output = allocate(16, 128 | 4);
    allocate(256, 32 | 8);
    allocate(256, 16 | 8);
    gpu.mgpuWriteRawBytesAt(
      uniform,
      Uint32List.fromList([10, 0, 0, 0]).buffer.asUint8List(),
      256,
    );
    gpu.mgpuWriteRawBytesAt(
      uniform,
      Uint32List.fromList([20, 0, 0, 0]).buffer.asUint8List(),
      512,
    );
    gpu.mgpuWriteRawBytesAt(
      source,
      Uint32List.fromList([1, 2, 3, 4]).buffer.asUint8List(),
      256,
    );
    shader = gpu.mgpuCreateComputeShader();
    gpu.mgpuLoadKernel(shader, '''
@group(0) @binding(0) var<uniform> params: vec4<u32>;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@compute @workgroup_size(1) fn main() {
  for (var i = 0u; i < 4u; i++) { dst[i] = params.x + src[i]; }
}
''');
    check(
      gpu.mgpuSetBufferRange(shader, 0, uniform, 2, 256, 16) == 0,
      'Uniform range refused',
    );
    check(
      gpu.mgpuSetBufferRange(shader, 1, source, 1, 256, 16) == 0,
      'Read-only range refused',
    );
    check(
      gpu.mgpuSetBufferRange(shader, 2, output, 0, 0, 16) == 0,
      'Storage output refused',
    );
    Future<void> verify(int base) async {
      await gpu.mgpuDispatch(shader!, 1, 1, 1);
      final result = Uint32List(4);
      await gpu.mgpuReadAsyncUint32(output, result, readElements: 4);
      for (var i = 0; i < 4; i++) {
        check(result[i] == base + i + 1, 'Unexpected compute output: $result');
      }
    }

    await verify(10);
    check(
      gpu.mgpuSetBufferRange(shader, 0, uniform, 2, 512, 16) == 0,
      'Same-buffer range change refused',
    );
    await verify(20);
    check(
      gpu.mgpuSetBufferRange(shader, 0, uniform, 2, 4, 16) == 1,
      'Unaligned uniform offset admitted',
    );
    check(
      gpu.mgpuSetBufferRange(shader, 0, source, 2, 0, 16) == 1,
      'Wrong usage admitted',
    );
    check(
      gpu.mgpuSetBufferRange(shader, 1, source, 1, 768, 16) == 1,
      'Out-of-range binding admitted',
    );
    check(
      gpu.mgpuCreateBufferWithUsage(6, 128) == null,
      'Malformed allocation admitted',
    );
  } catch (error, stack) {
    outcome = 'failed: $error\n$stack';
  } finally {
    try {
      if (shader != null) gpu.mgpuDestroyComputeShader(shader);
      for (final buffer in buffers.reversed) {
        gpu.mgpuDestroyBuffer(buffer);
      }
      if (initialized) await gpu.mgpuDestroyContext();
    } catch (error) {
      outcome = '$outcome; cleanup failed: $error';
    }
    globalContext.setProperty('bufferSmokeResult'.toJS, outcome.toJS);
  }
}

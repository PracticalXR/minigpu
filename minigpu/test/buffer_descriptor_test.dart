import 'dart:typed_data';

import 'package:minigpu/minigpu.dart';
import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';
import 'package:test/test.dart';

// Recording boundaries only: native/browser integration exercises real buffers.
class _LegacyPlatform extends MinigpuPlatform {
  int? allocatedBytes;
  @override
  PlatformBuffer createBuffer(int bufferSize, BufferDataType dataType) {
    allocatedBytes = bufferSize;
    expect(dataType, BufferDataType.uint32);
    return _Buffer();
  }

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _Buffer implements PlatformBuffer {
  var destroyed = 0;
  @override
  void destroy() {
    destroyed++;
  }

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _Shader extends PlatformComputeShader {
  final List<BufferBindingDescriptor> binds = [];
  var destroyed = 0;
  @override
  void setBufferRange(
    int slot,
    PlatformBuffer buffer,
    BufferBindingDescriptor descriptor,
  ) {
    binds.add(descriptor);
  }

  @override
  void destroy() {
    destroyed++;
  }

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

void main() {
  test('legacy platform keeps byte sizes and refuses unsupported usage', () {
    final platform = _LegacyPlatform();
    platform.createBufferFromDescriptor(
      BufferDescriptor(sizeBytes: 256, usage: BufferUsage.legacy),
    );
    expect(platform.allocatedBytes, 256);
    expect(
      () => platform.createBufferFromDescriptor(
        BufferDescriptor(sizeBytes: 256, usage: BufferUsage.uniform),
      ),
      throwsUnsupportedError,
    );
    expect(platform.allocatedBytes, 256);
    expect(
      () => BufferDescriptor(sizeBytes: 6, usage: BufferUsage.storage),
      throwsArgumentError,
    );
    expect(
      () => BufferDescriptor(sizeBytes: 256, usage: 0x100),
      throwsArgumentError,
    );
  });

  test(
    'typed API rejects incompatible ranges and use after disposal',
    () async {
      final rawBuffer = _Buffer();
      final buffer = Buffer(
        rawBuffer,
        null,
        BufferDescriptor(sizeBytes: 512, usage: BufferUsage.uniform),
      );
      final rawShader = _Shader();
      final shader = ComputeShader(rawShader);
      final uniform = BufferBindingDescriptor(
        kind: BufferBindingKind.uniform,
        offsetBytes: 256,
        sizeBytes: 16,
      );
      shader.setBufferRange(0, buffer, uniform);
      expect(rawShader.binds, [uniform]);
      expect(
        () => shader.setBufferRange(32, buffer, uniform),
        throwsArgumentError,
      );
      expect(
        () => shader.setBufferRange(
          0,
          buffer,
          BufferBindingDescriptor(
            kind: BufferBindingKind.storage,
            sizeBytes: 16,
          ),
        ),
        throwsArgumentError,
      );
      expect(
        () => shader.setBufferRange(
          0,
          buffer,
          BufferBindingDescriptor(
            kind: BufferBindingKind.uniform,
            offsetBytes: 512,
            sizeBytes: 16,
          ),
        ),
        throwsArgumentError,
      );
      await expectLater(buffer.read(Uint32List(4), 4), throwsUnsupportedError);
      await expectLater(
        buffer.writeRawBytes(Uint8List(16)),
        throwsUnsupportedError,
      );
      expect(rawShader.binds, [uniform]);
      buffer.destroy();
      expect(() => shader.setBufferRange(0, buffer, uniform), throwsStateError);
      shader.destroy();
      shader.destroy();
      expect(rawShader.destroyed, 1);
      expect(rawBuffer.destroyed, 1);
    },
  );
}

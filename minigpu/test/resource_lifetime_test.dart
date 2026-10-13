import 'dart:async';
import 'dart:typed_data';

import 'package:minigpu/minigpu.dart';
import 'package:minigpu/src/buffer.dart' show MinigpuNotInitializedError;
import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';
import 'package:test/test.dart';

// Exercises the real owning wrapper against an asynchronous platform boundary.
// Actual native/browser allocation and compute are qualified separately.
class _Platform extends MinigpuPlatform {
  final events = <String>[];
  Completer<void>? opening;
  Completer<void>? closing;
  bool failOpen = false;
  bool failBufferClose = false;
  bool failShaderClose = false;
  int serial = 0;

  @override
  Future<void> initializeContext() {
    events.add('open');
    if (failOpen) throw StateError('open failed synchronously');
    return opening?.future ?? Future<void>.value();
  }

  @override
  Future<void> destroyContext() {
    events.add('close');
    return closing?.future ?? Future<void>.value();
  }

  @override
  PlatformBuffer createBuffer(int bufferSize, BufferDataType dataType) =>
      _Buffer(this, 'buffer${serial++}');

  @override
  PlatformComputeShader createComputeShader() =>
      _Shader(this, 'shader${serial++}');

  @override
  PlatformTexture createTexture(TextureDescriptor descriptor) => _Texture(this);

  @override
  PlatformSampler createSampler(SamplerDescriptor descriptor) => _Sampler(this);
}

class _Texture extends PlatformTexture {
  _Texture(this.platform);
  final _Platform platform;
  @override
  PlatformTextureView createView(TextureViewDescriptor descriptor) =>
      _View(platform);
  @override
  void destroy() => platform.events.add('texture');
  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _View extends PlatformTextureView {
  _View(this.platform);
  final _Platform platform;
  @override
  void destroy() => platform.events.add('view');
}

class _Sampler extends PlatformSampler {
  _Sampler(this.platform);
  final _Platform platform;
  @override
  void destroy() => platform.events.add('sampler');
}

class _Buffer extends PlatformBuffer {
  _Buffer(this.platform, this.id);
  final _Platform platform;
  final String id;

  @override
  void destroy() {
    platform.events.add(id);
    if (platform.failBufferClose) throw StateError('buffer cleanup failed');
  }

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class _Shader extends PlatformComputeShader {
  _Shader(this.platform, this.id);
  final _Platform platform;
  final String id;

  @override
  void destroy() {
    platform.events.add(id);
    if (platform.failShaderClose) throw StateError('shader cleanup failed');
  }

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

void main() {
  test(
    'owned texture family retires before detach, tokens cannot retire twice',
    () async {
      final platform = _Platform();
      final gpu = Minigpu.withPlatform(platform);
      await gpu.init();
      final texture = gpu.createTexture(
        TextureDescriptor(
          width: 2,
          height: 2,
          format: TextureFormat.rgba8Unorm,
          usage: TextureUsage.copyDst | TextureUsage.textureBinding,
        ),
      );
      final view = texture.createView();
      final sampler = gpu.createSampler(SamplerDescriptor());
      final allocated = gpu.resourceAllocationSnapshot;
      expect(allocated.texturesCreated, 1);
      expect(allocated.textureViewsCreated, 1);
      expect(allocated.samplersCreated, 1);
      expect(allocated.totalRetired, 0);
      final token = sampler.retirement;
      await gpu.destroy();
      expect(platform.events, ['open', 'view', 'texture', 'sampler', 'close']);
      expect(texture.isValid, isFalse);
      expect(view.isValid, isFalse);
      expect(sampler.isValid, isFalse);
      final retired = gpu.resourceAllocationSnapshot;
      expect(retired.texturesRetired, 1);
      expect(retired.textureViewsRetired, 1);
      expect(retired.samplersRetired, 1);
      expect(retired.texturesLive, 0);
      expect(retired.textureViewsLive, 0);
      expect(retired.samplersLive, 0);
      token
          .retire(); // A queued finalizer may run after owning-context closure.
      texture.destroy();
      view.destroy();
      sampler.destroy();
      expect(platform.events.where((e) => e == 'sampler'), hasLength(1));
      expect(texture.createView, throwsStateError);
      expect(() => sampler.platformSampler, throwsStateError);
    },
  );

  test('close fences pending init and reopens only after retirement', () async {
    final platform = _Platform()
      ..opening = Completer<void>()
      ..closing = Completer<void>();
    final gpu = Minigpu.withPlatform(platform);
    final opened = gpu.init();
    final closed = gpu.destroy();
    expect(identical(closed, gpu.destroy()), isTrue);
    final reopened = gpu.init();
    expect(gpu.isInitialized, isFalse);
    expect(gpu.createComputeShader, throwsA(isA<MinigpuNotInitializedError>()));
    platform.opening!.complete();
    await opened;
    await Future<void>.delayed(Duration.zero);
    expect(platform.events, ['open', 'close']);
    expect(gpu.isInitialized, isFalse);
    platform.closing!.complete();
    await closed;
    await reopened;
    expect(platform.events, ['open', 'close', 'open']);
    expect(gpu.isInitialized, isTrue);
    await gpu.destroy();
  });

  test(
    'retained children retire once before detach and stay fenced on reopen',
    () async {
      final platform = _Platform()..closing = Completer<void>();
      final gpu = Minigpu.withPlatform(platform);
      await gpu.init();
      final buffer = gpu.createBuffer(16, BufferDataType.uint32);
      final firstShader = gpu.createComputeShader();
      final secondShader = gpu.createComputeShader();
      firstShader.destroy();
      firstShader.destroy();
      expect(gpu.liveShaderCount, 1);
      expect(gpu.liveBufferCount, 1);
      gpu.destroySync();
      final retired = gpu.destroy();
      expect(platform.events, [
        'open',
        'shader1',
        'shader2',
        'buffer0',
        'close',
      ]);
      expect(buffer.isValid, isFalse);
      expect(gpu.liveBufferCount, 0);
      expect(gpu.liveShaderCount, 0);
      final reopened = gpu.init();
      platform.closing!.complete();
      await retired;
      await reopened;
      await expectLater(buffer.writeRawBytes(Uint8List(4)), throwsStateError);
      expect(() => secondShader.platformShader, throwsStateError);
      expect(() => secondShader.dispatchFire(1, 1, 1), throwsStateError);
      expect(() => secondShader.setBufferAtSlot(0, buffer), throwsStateError);
      buffer.destroy();
      secondShader.destroy();
      expect(platform.events.where((e) => e == 'buffer0'), hasLength(1));
      final fresh = gpu.createBuffer(4, BufferDataType.uint32);
      expect(fresh.isValid, isTrue);
      await gpu.destroy();
    },
  );

  test(
    'cleanup failure attempts every child and detach, then stays closed',
    () async {
      final platform = _Platform()
        ..failBufferClose = true
        ..failShaderClose = true;
      final gpu = Minigpu.withPlatform(platform);
      await gpu.init();
      gpu.createBuffer(16, BufferDataType.uint32);
      gpu.createComputeShader();
      final retired = gpu.destroy();
      await expectLater(retired, throwsStateError);
      expect(platform.events, ['open', 'shader1', 'buffer0', 'close']);
      expect(gpu.isInitialized, isFalse);
      expect(gpu.liveBufferCount, 0);
      expect(gpu.liveShaderCount, 0);
      expect(identical(retired, gpu.destroy()), isTrue);
      await expectLater(gpu.init(), throwsStateError);
      expect(platform.events.where((e) => e == 'open'), hasLength(1));
    },
  );

  test(
    'synchronous initialization failure does not poison the init flight',
    () async {
      final platform = _Platform()..failOpen = true;
      final gpu = Minigpu.withPlatform(platform);
      await expectLater(gpu.init(), throwsStateError);
      platform.failOpen = false;
      await gpu.init();
      expect(gpu.isInitialized, isTrue);
      expect(platform.events, ['open', 'open']);
      await gpu.destroy();
    },
  );
}

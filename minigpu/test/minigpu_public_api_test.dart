// Public-layer contracts against a fake platform: dataType inference, the
// copyBuffer arguments and params, the rebind sentinel and buffer ownership,
// teardown and recovery after a failed cleanup, finalizer collection, texture
// retirement, and the shader warmer's bindings, residency and loss
// classification. Real-device copies: minigpu_copy_buffer_edges_test.dart.

import 'dart:async';
import 'dart:typed_data';

import 'package:minigpu/minigpu.dart';
import 'package:minigpu/src/buffer.dart' show MinigpuNotInitializedError;
import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';
import 'package:test/test.dart';

typedef _Transfer = ({TypedData data, BufferDataType type});
typedef _Bind = ({
  int slot,
  String buffer,
  BufferBindingKind? kind,
  int? bytes,
});
typedef _Dispatch = ({
  List<int> groups,
  String? dst,
  _Buf? paramsBuffer,
  List<int>? params,
});

class _Gpu extends MinigpuPlatform {
  final events = <String>[];
  final buffers = <_Buf>[];
  final created = <({int bytes, BufferDataType? type, int? usage})>[];
  final shaders = <_Shader>[];
  final dispatches = <_Dispatch>[];
  int builds = 0;
  Completer<void>? opening;
  Future<void> Function()? onDispatch;
  Future<void> Function()? onBuild;
  bool failBufferDestroy = false;
  void Function()? lost;
  int serial = 0;

  @override
  Future<void> initializeContext() async {
    events.add('open');
    await opening?.future;
  }

  @override
  Future<void> destroyContext() async {
    events.add('close');
  }

  @override
  void setDeviceLostCallback(void Function()? callback) => lost = callback;

  void lose() => lost!();

  @override
  PlatformBuffer createBuffer(int bufferSize, BufferDataType dataType) =>
      _buffer(bufferSize, dataType, null);

  @override
  PlatformBuffer createBufferFromDescriptor(BufferDescriptor descriptor) =>
      _buffer(descriptor.sizeBytes, null, descriptor.usage);

  _Buf _buffer(int bytes, BufferDataType? type, int? usage) {
    final buffer = _Buf(this, 'buffer${serial++}');
    buffers.add(buffer);
    created.add((bytes: bytes, type: type, usage: usage));
    return buffer;
  }

  @override
  PlatformComputeShader createComputeShader() {
    final shader = _Shader(this, 'shader${serial++}');
    shaders.add(shader);
    return shader;
  }

  @override
  PlatformVideoTexture? importVideoFrame(ExternalVideoBuffer buf) =>
      _Video(this, 'video${serial++}');

  @override
  PlatformSharedOutputTexture? createSharedOutputTexture(
    int width,
    int height,
  ) => _Shared(this, 'shared${serial++}');
}

/// A backend that can lose its device and be replaced on the same adapter.
class _RecoverableGpu extends _Gpu {
  bool live = false;
  int starts = 0, detaches = 0;
  Completer<void> finalContext = Completer<void>();

  @override
  DeviceAdapterIdentity? get creationAdapterIdentity =>
      const DeviceAdapterIdentity(backend: 1, kind: 1, low: 27, high: 0);

  @override
  Future<void> initializeContext() async {
    live = true;
    starts++;
  }

  @override
  Future<void> initializeContextForRecovery(
    DeviceAdapterIdentity expected,
  ) async {
    finalContext = Completer<void>();
    await initializeContext();
  }

  @override
  Future<void> observeContextRetirement({required int deviceGeneration}) =>
      finalContext.future;

  @override
  Future<void> destroyContext() async {
    detaches++;
    live = false;
    finalContext.complete();
  }

  @override
  DeviceBufferLimits? get currentDeviceBufferLimits => !live
      ? null
      : DeviceBufferLimits(
          deviceGeneration: starts,
          maxBufferSize: 1024,
          maxStorageBufferBindingSize: 1024,
        );

  @override
  void lose() {
    live = false;
    super.lose();
  }
}

class _Buf extends PlatformBuffer {
  _Buf(this.gpu, this.id);
  final _Gpu gpu;
  final String id;
  final writes = <_Transfer>[];
  final reads = <_Transfer>[];
  bool destroyed = false;

  @override
  Future<void> write(
    TypedData inputData,
    int size, {
    BufferDataType dataType = BufferDataType.float32,
  }) async {
    writes.add((data: inputData, type: dataType));
  }

  @override
  Future<void> read(
    TypedData outputData,
    int readElements, {
    int elementOffset = 0,
    int readBytes = 0,
    int byteOffset = 0,
    BufferDataType dataType = BufferDataType.float32,
  }) async {
    reads.add((data: outputData, type: dataType));
    if (outputData is Float32List) {
      for (var i = 0; i < outputData.length; i++) {
        outputData[i] = i + 1.0;
      }
    }
  }

  @override
  void destroy() {
    gpu.events.add(id);
    destroyed = true;
    if (gpu.failBufferDestroy) throw StateError('buffer cleanup failed');
  }
}

class _Shader extends PlatformComputeShader {
  _Shader(this.gpu, this.id);
  final _Gpu gpu;
  final String id;
  final binds = <_Bind>[];
  final bound = <int, _Buf>{};
  String? source;
  bool destroyed = false;

  @override
  void loadKernelString(String kernelString) {
    source = kernelString;
  }

  @override
  bool hasKernel() => source != null;

  @override
  void setBuffer(int tag, PlatformBuffer buffer) {
    final native = buffer as _Buf;
    bound[tag] = native;
    binds.add((slot: tag, buffer: native.id, kind: null, bytes: null));
  }

  @override
  void setBufferRange(
    int slot,
    PlatformBuffer buffer,
    BufferBindingDescriptor descriptor,
  ) {
    final native = buffer as _Buf;
    bound[slot] = native;
    binds.add((
      slot: slot,
      buffer: native.id,
      kind: descriptor.kind,
      bytes: descriptor.sizeBytes,
    ));
  }

  /// Records what the dispatch would read: copyBuffer binds dst at slot 1
  /// and its params at slot 2.
  @override
  Future<void> dispatch(int groupsX, int groupsY, int groupsZ) {
    final params = bound[2];
    final last = params == null || params.writes.isEmpty
        ? null
        : params.writes.last.data;
    gpu.dispatches.add((
      groups: [groupsX, groupsY, groupsZ],
      dst: bound[1]?.id,
      paramsBuffer: params,
      params: last is Uint32List ? List<int>.of(last) : null,
    ));
    return gpu.onDispatch?.call() ?? Future<void>.value();
  }

  /// Compiles only: a build never reaches [dispatch].
  @override
  Future<void> buildPipeline() {
    gpu.builds++;
    return gpu.onBuild?.call() ?? Future<void>.value();
  }

  @override
  void destroy() {
    gpu.events.add(id);
    destroyed = true;
  }
}

class _Video extends PlatformVideoTexture {
  _Video(this.gpu, this.id);
  final _Gpu gpu;
  final String id;

  @override
  int get numPlanes => 1;
  @override
  int get width => 2;
  @override
  int get height => 2;
  @override
  ExternalPixelFormat get pixelFormat => ExternalPixelFormat.rgba32;
  @override
  void setOnShader(PlatformComputeShader shader, int slot, int planeIndex) {}
  @override
  PlatformBuffer toRGBA() => gpu._buffer(16, BufferDataType.uint32, null);
  @override
  void destroy() => gpu.events.add(id);
}

class _Shared extends PlatformSharedOutputTexture {
  _Shared(this.gpu, this.id);
  final _Gpu gpu;
  final String id;

  @override
  int get width => 2;
  @override
  int get height => 2;
  @override
  int get d3d11Handle => 7;
  @override
  bool copyFromBuffer(PlatformBuffer src) => true;
  @override
  void destroy() => gpu.events.add(id);
}

const _kAdd = '''
@group(0) @binding(0) var<storage, read_write> b : array<f32>;
@compute @workgroup_size(1)
fn main() {
  b[0] = b[0] + 1.0;
}
''';

const _kScale = '''
struct Params { n : u32, scale : f32 }
@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read> input : array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;
@compute @workgroup_size(1)
fn main() {
  output[0] = input[0] * params.scale;
}
''';

Future<(_Gpu, Minigpu)> _open([_Gpu? backend]) async {
  final platform = backend ?? _Gpu();
  final gpu = Minigpu.withPlatform(platform);
  await gpu.init();
  return (platform, gpu);
}

/// Creates a shader and keeps no reference to it.
@pragma('vm:never-inline')
void _dropShader(Minigpu gpu) {
  gpu.createComputeShader();
}

/// Makes garbage until [done] holds: a dropped wrapper needs a collection and
/// then an event-loop turn for its finalizer. The large list lands in old
/// space and the small ones in new space, so both generations collect.
Future<void> _collectUntil(bool Function() done) async {
  final deadline = DateTime.now().add(const Duration(seconds: 30));
  final keep = <Object>[];
  var rounds = 0;
  while (!done()) {
    if (DateTime.now().isAfter(deadline)) {
      fail('not collected after $rounds rounds of garbage');
    }
    keep
      ..clear()
      ..add(Uint8List(1 << 20))
      ..add(List<Object>.generate(20000, (i) => [i]));
    rounds++;
    await Future<void>.delayed(Duration.zero);
  }
  keep.clear();
}

void main() {
  group('Buffer dataType', () {
    test('read and write infer it from the list', () async {
      final (platform, gpu) = await _open();
      final buffer = gpu.createBuffer(64, BufferDataType.float32);
      final native = platform.buffers.single;

      await buffer.write(Uint8List(4), 4);
      await buffer.write(Int16List(2), 2);
      await buffer.write(Int64List(1), 1);
      await buffer.write(Uint16List(2), 2, dataType: BufferDataType.float16);
      await buffer.write(ByteData(8), 2, dataType: BufferDataType.uint32);
      await buffer.write(Uint8ClampedList.fromList([1, 2, 3, 4]), 4);
      expect(native.writes.map((w) => w.type), [
        BufferDataType.uint8,
        BufferDataType.int16,
        BufferDataType.int64,
        BufferDataType.float16,
        BufferDataType.uint32,
        BufferDataType.uint8,
      ]);
      // A clamped list reaches the backend as the plain list of its bytes.
      expect(native.writes.last.data, isA<Uint8List>());
      expect((native.writes.last.data as Uint8List).toList(), [1, 2, 3, 4]);

      // A read lands in the caller's list, also through a SIMD list's view.
      final quad = Float32x4List(1);
      await buffer.read(quad, 4);
      expect(native.reads.last.type, BufferDataType.float32);
      expect([quad[0].x, quad[0].y, quad[0].z, quad[0].w], [1.0, 2.0, 3.0, 4.0]);
      await buffer.read(Int32List(2), 2);
      expect(native.reads.last.type, BufferDataType.int32);
    });

    test('a type that does not match the list never reaches the backend',
        () async {
      final (platform, gpu) = await _open();
      final buffer = gpu.createBuffer(64, BufferDataType.float32);
      final native = platform.buffers.single;

      await expectLater(
        buffer.write(Float32List(4), 4, dataType: BufferDataType.uint32),
        throwsArgumentError,
      );
      await expectLater(
        buffer.write(Int8List(4), 4, dataType: BufferDataType.uint8),
        throwsArgumentError,
      );
      await expectLater(
        buffer.read(Uint8List(4), 4, dataType: BufferDataType.float32),
        throwsArgumentError,
      );
      // Raw bytes have no element type to infer.
      await expectLater(buffer.write(ByteData(8), 2), throwsArgumentError);
      expect(native.writes, isEmpty);
      expect(native.reads, isEmpty);

      // A destroyed buffer says so first.
      buffer.destroy();
      await expectLater(
        buffer.write(Float32List(4), 4, dataType: BufferDataType.uint32),
        throwsStateError,
      );
    });
  });

  group('copyBuffer', () {
    test('counts src elements in bytes and folds past 65535 groups', () async {
      final (platform, gpu) = await _open();
      const wide = 65535 * 64 + 1; // one word past a full dispatch row
      final pairs = [
        (BufferDataType.uint8, 16, 7),
        (BufferDataType.float64, 32, 3),
        (BufferDataType.float32, wide * 4, wide),
      ];
      for (final (type, bytes, count) in pairs) {
        await gpu.copyBuffer(
          gpu.createBuffer(bytes, type),
          gpu.createBuffer(bytes, type),
          elementCount: count,
        );
      }
      expect(platform.dispatches.map((d) => d.groups), [
        [1, 1, 1],
        [1, 1, 1],
        [65535, 2, 1],
      ]);
      expect(platform.dispatches.map((d) => d.params), [
        [1, 3, 64, 0], // one whole word, then the low 3 bytes of the next
        [6, 0, 64, 0],
        [wide, 0, 65535 * 64, 0],
      ]);
      expect(platform.shaders, hasLength(1));
    });

    test('refuses bad counts and destroyed buffers before any bind', () async {
      final (platform, gpu) = await _open();
      final src = gpu.createBuffer(64, BufferDataType.float32);
      final small = gpu.createBuffer(32, BufferDataType.float32);
      final dead = gpu.createBuffer(64, BufferDataType.float32)..destroy();

      await expectLater(
        gpu.copyBuffer(src, small, elementCount: 9),
        throwsRangeError,
      );
      await expectLater(
        gpu.copyBuffer(small, src, elementCount: 9),
        throwsRangeError,
      );
      await expectLater(
        gpu.copyBuffer(src, small, elementCount: -1),
        throwsRangeError,
      );
      await expectLater(
        gpu.copyBuffer(src, dead, elementCount: 1),
        throwsStateError,
      );
      await expectLater(
        gpu.copyBuffer(dead, src, elementCount: 1),
        throwsStateError,
      );
      // Self-copy and zero bytes are no-ops.
      await gpu.copyBuffer(src, src, elementCount: 16);
      await gpu.copyBuffer(src, small, elementCount: 0);
      expect(platform.shaders, isEmpty);
      expect(platform.dispatches, isEmpty);
    });

    test('copies in flight own their params; finished ones are reused',
        () async {
      final (platform, gpu) = await _open();
      final src = gpu.createBuffer(64, BufferDataType.float32);
      final a = gpu.createBuffer(64, BufferDataType.float32);
      final b = gpu.createBuffer(64, BufferDataType.float32);
      final gate = Completer<void>();
      platform.onDispatch = () => gate.future;

      final first = gpu.copyBuffer(src, a, elementCount: 5);
      final second = gpu.copyBuffer(src, b, elementCount: 9);
      final [one, two] = platform.dispatches;
      expect([one.dst, two.dst], [platform.buffers[1].id, platform.buffers[2].id]);
      expect(one.params, [5, 0, 64, 0]);
      expect(two.params, [9, 0, 64, 0]);
      expect(one.paramsBuffer, isNot(same(two.paramsBuffer)));
      gate.complete();
      await Future.wait([first, second]);

      final made = platform.buffers.length;
      await gpu.copyBuffer(src, a, elementCount: 3);
      expect(platform.buffers, hasLength(made));
      expect(platform.dispatches.last.params, [3, 0, 64, 0]);
      // Params buffers are the context's, not the caller's.
      expect(gpu.liveBufferCount, 3);
      expect(gpu.resourceAllocationSnapshot.buffersCreated, 3);

      await gpu.destroy();
      expect(one.paramsBuffer!.destroyed, isTrue);
      expect(two.paramsBuffer!.destroyed, isTrue);
      await expectLater(
        gpu.copyBuffer(src, a, elementCount: 1),
        throwsA(isA<MinigpuNotInitializedError>()),
      );
    });
  });

  group('buffer bookkeeping', () {
    test('the rebind sentinel is not a live, counted or owned buffer',
        () async {
      final (platform, gpu) = await _open();
      final shader = gpu.createComputeShader();
      final a = gpu.createBuffer(16, BufferDataType.float32);
      shader.setBufferAtSlot(0, a);
      a.destroy();
      final b = gpu.createBuffer(16, BufferDataType.float32);
      shader.setBufferAtSlot(0, b);

      final sentinel = gpu.rebindSentinel;
      final [na, nb, ns] = platform.buffers;
      expect(sentinel.platformBuffer, same(ns));
      // Rebinding over a destroyed buffer passes through the sentinel.
      expect(platform.shaders.single.binds.map((bind) => bind.buffer), [
        na.id,
        ns.id,
        nb.id,
      ]);
      expect(gpu.liveBufferCount, 1);
      expect(gpu.resourceAllocationSnapshot.buffersCreated, 2);
      expect(gpu.ownsBuffer(sentinel), isFalse);
      expect(gpu.rebindSentinel, same(sentinel));

      await gpu.destroy();
      expect(sentinel.isValid, isFalse);
      expect(ns.destroyed, isTrue);
    });

    test('ownsBuffer is per context and per device epoch', () async {
      final (_, gpu) = await _open();
      final (_, other) = await _open();
      final buffer = gpu.createBuffer(16, BufferDataType.float32);
      final destroyed = gpu.createBuffer(16, BufferDataType.float32)..destroy();
      expect(gpu.ownsBuffer(buffer), isTrue);
      expect(other.ownsBuffer(buffer), isFalse);
      expect(gpu.ownsBuffer(destroyed), isFalse);

      await gpu.destroy();
      expect(gpu.ownsBuffer(buffer), isFalse);
      await gpu.init();
      expect(gpu.ownsBuffer(gpu.createBuffer(16, BufferDataType.float32)),
          isTrue);
      await other.destroy();
    });

    test('setBuffer keeps the type of a bind failure', () async {
      final (_, gpu) = await _open();
      final shader = gpu.createComputeShader();
      final buffer = gpu.createBuffer(16, BufferDataType.float32);
      final dead = gpu.createBuffer(16, BufferDataType.float32)..destroy();
      expect(() => shader.setBuffer('dead', dead), throwsStateError);
      shader.destroy();
      expect(() => shader.setBuffer('live', buffer), throwsStateError);
    });

    test('video and shared output textures retire with the context', () async {
      final (platform, gpu) = await _open();
      final video = gpu.importVideoFrame(
        const ExternalVideoBuffer(
          contentType: ExternalContentType.cpu,
          pixelFormat: ExternalPixelFormat.rgba32,
          width: 2,
          height: 2,
          planes: [],
        ),
      )!;
      final shared = gpu.createSharedOutputTexture(2, 2)!;
      final rgba = video.toRGBA();
      expect(gpu.liveBufferCount, 1);
      expect(gpu.ownsBuffer(rgba), isTrue);

      final dead = gpu.createBuffer(16, BufferDataType.uint32)..destroy();
      expect(() => shared.copyFromBuffer(dead), throwsStateError);
      expect(() => shared.copyFromBufferAsync(dead), throwsStateError);
      expect(() => shared.copyFromBufferF32Async(dead), throwsStateError);
      expect(shared.copyFromBuffer(rgba), isTrue);

      platform.events.clear();
      await gpu.destroy();
      expect(platform.events, ['video0', 'shared1', 'buffer2', 'close']);
      expect(video.isValid, isFalse);
      expect(shared.isValid, isFalse);
      expect(rgba.isValid, isFalse);
      expect(() => video.width, throwsStateError);
      expect(() => shared.d3d11Handle, throwsStateError);
    });
  });

  group('teardown', () {
    test('a failed in-flight init torn down by destroySync does not wedge init',
        () async {
      final platform = _Gpu()..opening = Completer<void>();
      final gpu = Minigpu.withPlatform(platform);
      final opening = expectLater(gpu.init(), throwsStateError);
      gpu.destroySync();
      // destroy() callers still get the teardown's error.
      final closing = expectLater(gpu.destroy(), throwsStateError);
      platform.opening!.completeError(StateError('open failed'));
      platform.opening = null;
      await opening;
      await closing;

      await gpu.init();
      expect(gpu.isInitialized, isTrue);
      expect(platform.events, ['open', 'close', 'open']);
      expect(gpu.createBuffer(16, BufferDataType.float32).isValid, isTrue);
      await gpu.destroy();
    });

    test('recovery goes on when a child fails to destroy but the context '
        'was released', () async {
      final platform = _RecoverableGpu();
      final gpu = Minigpu.withPlatform(platform);
      await gpu.init();
      final epoch = gpu.deviceEpoch;
      final child = gpu.createBuffer(16, BufferDataType.float32);
      platform.failBufferDestroy = true;
      platform.lose();

      // The child's error is reported as uncaught; bridge the result out of
      // the zone that catches it.
      final uncaught = <Object>[];
      final recovered = Completer<int>();
      runZonedGuarded(() {
        gpu
            .recoverLostDevice(deviceEpoch: epoch)
            .then(recovered.complete, onError: recovered.completeError);
      }, (error, _) => uncaught.add(error));

      expect(await recovered.future, epoch + 1);
      expect(
        uncaught.single,
        isA<StateError>().having(
          (e) => e.message,
          'message',
          'buffer cleanup failed',
        ),
      );
      expect(gpu.recoveryStatus.phase, GpuRecoveryPhase.ready);
      expect(gpu.isInitialized, isTrue);
      expect(platform.starts, 2);
      expect(child.isValid, isFalse);

      platform.failBufferDestroy = false;
      await gpu.destroy();
      expect(platform.detaches, 2);
    });

    test(
      'a dropped shader is collected and its finalizer releases it',
      () async {
        final (platform, gpu) = await _open();
        _dropShader(gpu);
        expect(gpu.liveShaderCount, 1);
        await _collectUntil(() => gpu.liveShaderCount == 0);
        final native = platform.shaders.single;
        expect(native.destroyed, isTrue);
        expect(gpu.resourceAllocationSnapshot.computeShadersRetired, 1);
        await gpu.destroy();
        expect(platform.events.where((e) => e == native.id), hasLength(1));
      },
      timeout: const Timeout(Duration(seconds: 60)),
    );
  });

  group('shader warmer', () {
    test('buildPipeline compiles without dispatching; a destroyed shader or '
        'binding refuses', () async {
      final (platform, gpu) = await _open();
      final shader = gpu.createComputeShader()..loadKernelString(_kAdd);
      final buffer = gpu.createBuffer(16, BufferDataType.float32);
      shader.setBuffer('b', buffer);
      await shader.buildPipeline();
      expect(platform.builds, 1);
      expect(platform.dispatches, isEmpty);

      buffer.destroy();
      await expectLater(shader.buildPipeline(), throwsStateError);
      shader.destroy();
      await expectLater(shader.buildPipeline(), throwsStateError);
      expect(platform.builds, 1);
      await gpu.destroy();
    });

    test('binds each buffer as its declared kind, keeps the shader per epoch',
        () async {
      final (platform, gpu) = await _open();
      const spec = KernelWarmSpec(
        label: 'scale',
        source: _kScale,
        buffers: {'params': 2, 'input': 64, 'output': 100},
      );
      expect((await warmShaders(gpu, [spec]).done).phase, WarmPhase.ready);
      // Compiled, never run.
      expect(platform.builds, 1);
      expect(platform.dispatches, isEmpty);
      expect(platform.created, [
        (bytes: 256, type: null, usage: BufferUsage.uniform),
        (bytes: 256, type: BufferDataType.float32, usage: null),
        (bytes: 400, type: BufferDataType.float32, usage: null),
      ]);
      final shader = platform.shaders.single;
      expect(shader.binds.map((b) => (b.slot, b.kind, b.bytes)), [
        (0, BufferBindingKind.uniform, 256),
        (1, BufferBindingKind.readOnlyStorage, 256),
        (2, null, null),
      ]);
      expect(platform.buffers.every((b) => b.destroyed), isTrue);
      expect(gpu.liveBufferCount, 0);
      expect(gpu.liveShaderCount, 1);
      expect(shader.destroyed, isFalse);

      // Built in this epoch already: nothing to do.
      expect((await warmShaders(gpu, [spec]).done).phase, WarmPhase.ready);
      expect(platform.shaders, hasLength(1));

      // The kept shader goes with its context; a new epoch builds again.
      await gpu.destroy();
      expect(shader.destroyed, isTrue);
      await gpu.init();
      expect((await warmShaders(gpu, [spec]).done).phase, WarmPhase.ready);
      expect(platform.shaders, hasLength(2));
      expect(gpu.liveShaderCount, 1);
    });

    test('a failed build on a healthy device is a dispatch error, whatever '
        'its text', () async {
      final (platform, gpu) = await _open();
      platform.onBuild =
          () => Future.error(StateError('device lost while building'));
      final progress = await warmShaders(gpu, const [
        KernelWarmSpec(label: 'add', source: _kAdd, buffers: {'b': 1}),
        KernelWarmSpec(
          label: 'scale',
          source: _kScale,
          buffers: {'params': 2, 'input': 1, 'output': 1},
        ),
      ]).done;
      expect(progress.phase, WarmPhase.failed);
      expect(progress.completed, 2);
      expect(progress.errors.map((e) => e.stage), [
        WarmStage.dispatch,
        WarmStage.dispatch,
      ]);
      expect(gpu.liveShaderCount, 0);
      expect(gpu.liveBufferCount, 0);
    });

    test('a device lost during a build is deviceLost and ends the set',
        () async {
      final (platform, gpu) = await _open();
      platform.onBuild = () {
        platform.lose();
        return Future.error(StateError('late'));
      };
      final progress = await warmShaders(gpu, const [
        KernelWarmSpec(label: 'first', source: _kAdd, buffers: {'b': 1}),
        KernelWarmSpec(
          label: 'second',
          source: _kScale,
          buffers: {'params': 2, 'input': 1, 'output': 1},
        ),
      ]).done;
      expect(progress.phase, WarmPhase.failed);
      expect(progress.completed, 1);
      final error = progress.errors.single;
      expect(error.label, 'first');
      expect(error.stage, WarmStage.deviceLost);
      expect(error.message, contains('late'));
      expect(platform.shaders, hasLength(1));
      expect(gpu.liveShaderCount, 0);
    });
  });
}

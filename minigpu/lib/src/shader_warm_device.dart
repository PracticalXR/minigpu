part of 'shader_warmer.dart';

final _warmDevices = Expando<_WarmDevice>('shader warmer per device');

/// Scratch floor in f32 elements (256 bytes): a fixed-size struct binding
/// must fit whole at dispatch, and most parameter structs are smaller.
const _minScratchElements = 64;

/// WebGPU's default maxUniformBufferBindingSize.
const _maxUniformBytes = 65536;

final _bufferBinding = RegExp(
  r'@binding\s*\(\s*(\d+)\s*\)[^;{}]*?\bvar\s*<\s*(uniform|storage)\s*'
  r'(?:,\s*(read_write|read)\s*)?>',
);

/// The buffer kind each `@binding` of [source] declares; texture and sampler
/// bindings are left out.
Map<int, BufferBindingKind> _bufferKinds(String source) => {
  for (final match in _bufferBinding.allMatches(source))
    int.parse(match[1]!): match[2] == 'uniform'
        ? BufferBindingKind.uniform
        : match[3] == 'read_write'
        ? BufferBindingKind.storage
        : BufferBindingKind.readOnlyStorage, // WGSL storage defaults to read
};

/// All warmer handles targeting one device share a serial warm lane and one
/// residency set. A new device epoch invalidates both.
class _WarmDevice {
  _WarmDevice(this.gpu) {
    // Listeners outlive teardown; this one lives exactly as long as [gpu].
    gpu.onDeviceLost((lost) => lostEpoch = lost);
  }
  final Minigpu gpu;

  /// The shader that built each key, kept until its epoch ends: the backend
  /// shares a live pipeline with a real shader built from the same source and
  /// bindings, and frees it with the last shader that uses it.
  final Map<String, ComputeShader> built = {};
  Future<void> tail = Future<void>.value();
  int epoch = -1;
  int? lostEpoch;

  Future<WarmError?> warm(KernelWarmSpec spec) {
    if (epoch != gpu.deviceEpoch) {
      epoch = gpu.deviceEpoch;
      _release();
      tail = Future<void>.value();
    }
    final answer = Completer<WarmError?>();
    final expectedEpoch = epoch;
    tail = tail.then(
      (_) => _run(spec, answer, expectedEpoch),
      onError: (Object _, StackTrace __) => _run(spec, answer, expectedEpoch),
    );
    return answer.future;
  }

  Future<void> _run(
    KernelWarmSpec spec,
    Completer<WarmError?> answer,
    int expectedEpoch,
  ) async {
    final key = jsonEncode([
      spec.source,
      [
        for (final entry in spec.buffers.entries) [entry.key, entry.value],
      ],
    ]);
    if (!gpu.isInitialized) {
      answer.complete(WarmError(spec.label, WarmStage.create,
          'GPU device is not initialized'));
      return;
    }
    if (gpu.deviceEpoch != expectedEpoch || lostEpoch == expectedEpoch) {
      answer.complete(
        WarmError(
          spec.label,
          WarmStage.deviceLost,
          'GPU device changed before warm-up',
        ),
      );
      return;
    }
    if (built[key]?.isValid ?? false) {
      answer.complete(null);
      return;
    }
    final kinds = _bufferKinds(spec.source);
    final scratch = <Buffer>[];
    ComputeShader? shader;
    var keep = false;
    try {
      shader = gpu.createComputeShader();
      shader.loadKernelString(spec.source);
      // Tags take slots in first-seen order, as setBuffer assigns them, each
      // bound with the kind its @binding declares (setBuffer binds storage).
      var slot = 0;
      for (final count in spec.buffers.values) {
        final kind = kinds[slot] ?? BufferBindingKind.storage;
        // Counts are f32 elements; buffers take byte lengths.
        var bytes =
            (count < _minScratchElements ? _minScratchElements : count) * 4;
        final Buffer buffer;
        if (kind == BufferBindingKind.uniform) {
          if (bytes > _maxUniformBytes) bytes = _maxUniformBytes;
          buffer = gpu.createBufferFromDescriptor(
            BufferDescriptor(sizeBytes: bytes, usage: BufferUsage.uniform),
          );
        } else {
          buffer = gpu.createBuffer(bytes, BufferDataType.float32);
        }
        scratch.add(buffer);
        if (kind == BufferBindingKind.storage) {
          shader.setBufferAtSlot(slot, buffer);
        } else {
          shader.setBufferRange(
            slot,
            buffer,
            BufferBindingDescriptor(kind: kind, sizeBytes: bytes),
          );
        }
        slot++;
      }
      // Compiles without running the kernel. The build's completion, not the
      // timeout, owns the scratch lifetime.
      final build = shader.buildPipeline();
      try {
        await build.timeout(spec.timeout);
        if (_lost(expectedEpoch)) {
          answer.complete(
            WarmError(
              spec.label,
              WarmStage.deviceLost,
              'GPU device changed during warm-up',
            ),
          );
        } else {
          built[key] = shader;
          keep = true;
          answer.complete(null);
        }
      } on TimeoutException {
        answer.complete(
          WarmError(
            spec.label,
            WarmStage.timeout,
            'did not finish within ${spec.timeout.inSeconds}s',
          ),
        );
        // Keep the lane and scratch alive until the backend actually retires.
        try {
          await build;
        } catch (_) {
          /* Timeout remains the cause. */
        }
      }
    } catch (error) {
      if (!answer.isCompleted) {
        final lost = _lost(expectedEpoch);
        if (lost) _release();
        answer.complete(
          WarmError(
            spec.label,
            lost ? WarmStage.deviceLost : WarmStage.dispatch,
            '$error',
          ),
        );
      }
    } finally {
      for (final buffer in scratch) {
        try {
          buffer.destroy();
        } catch (_) {}
      }
      if (!keep) {
        try {
          shader?.destroy();
        } catch (_) {}
      }
    }
  }

  /// Loss by device state, not by error text.
  bool _lost(int expectedEpoch) =>
      !gpu.isInitialized ||
      gpu.deviceEpoch != expectedEpoch ||
      lostEpoch == expectedEpoch;

  void _release() {
    for (final shader in built.values) {
      try {
        shader.destroy();
      } catch (_) {}
    }
    built.clear();
  }
}

part of 'shader_warmer.dart';

final _warmDevices = Expando<_WarmDevice>('shader warmer per device');

/// All warmer handles targeting one device share a serial warm lane and one
/// residency set. A new device epoch invalidates both.
class _WarmDevice {
  _WarmDevice(this.gpu);
  final Minigpu gpu;
  final Set<String> built = {};
  Future<void> tail = Future<void>.value();
  int epoch = -1;

  Future<WarmError?> warm(KernelWarmSpec spec) {
    if (epoch != gpu.deviceEpoch) {
      epoch = gpu.deviceEpoch;
      built.clear();
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
    if (gpu.deviceEpoch != expectedEpoch) {
      answer.complete(
        WarmError(
          spec.label,
          WarmStage.deviceLost,
          'GPU device changed before warm-up',
        ),
      );
      return;
    }
    if (built.contains(key)) {
      answer.complete(null);
      return;
    }
    final scratch = <Buffer>[];
    ComputeShader? shader;
    try {
      shader = gpu.createComputeShader();
      shader.loadKernelString(spec.source);
      for (final entry in spec.buffers.entries) {
        // Counts are f32 elements; createBuffer requires byte lengths.
        final buffer = gpu.createBuffer(
          (entry.value <= 0 ? 1 : entry.value) * 4,
          BufferDataType.float32,
        );
        scratch.add(buffer);
        shader.setBuffer(entry.key, buffer);
      }
      // Backend completion, not timeout publication, owns scratch lifetime.
      final dispatch = shader.dispatch(1, 1, 1);
      try {
        await dispatch.timeout(spec.timeout);
        if (!gpu.isInitialized || gpu.deviceEpoch != expectedEpoch) {
          answer.complete(
            WarmError(
              spec.label,
              WarmStage.deviceLost,
              'GPU device changed during warm-up',
            ),
          );
        } else {
          built.add(key);
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
          await dispatch;
        } catch (_) {
          /* Timeout remains the cause. */
        }
      }
    } catch (error) {
      if (!answer.isCompleted) {
        final message = '$error';
        final lost =
            message.toLowerCase().contains('device lost') ||
            message.toLowerCase().contains('devicelost');
        if (lost) built.clear();
        answer.complete(
          WarmError(
            spec.label,
            lost ? WarmStage.deviceLost : WarmStage.dispatch,
            message,
          ),
        );
      }
    } finally {
      for (final buffer in scratch) {
        try {
          buffer.destroy();
        } catch (_) {}
      }
      try {
        shader?.destroy();
      } catch (_) {}
    }
  }
}

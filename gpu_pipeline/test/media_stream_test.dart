import 'package:gpu_pipeline/gpu_pipeline.dart';
import 'package:test/test.dart';

/// A stream outside the library that implements only the capture hooks.
/// Before they were protected, start() called a private method that no
/// subclass in another library could override.
class _CountingStream extends MediaStream {
  _CountingStream()
    : super(
        id: 'counting',
        deviceId: 'none',
        name: 'Counting',
        color: 0,
        type: MediaStreamType.audio,
        context: null,
      );

  int starts = 0;
  int stops = 0;

  @override
  Future<void> startCapture() async {
    starts++;
  }

  @override
  Future<void> stopCapture() async {
    stops++;
  }
}

/// A stream that pushes frames and has nothing to open.
class _PushOnlyStream extends MediaStream {
  _PushOnlyStream()
    : super(
        id: 'push_only',
        deviceId: 'none',
        name: 'Push only',
        color: 0,
        type: MediaStreamType.video,
        context: null,
      );
}

void main() {
  test('start and stop call the capture hooks once each', () async {
    final stream = _CountingStream();
    expect(stream.isActive, isFalse);

    await stream.start();
    await stream.start();
    expect(stream.isActive, isTrue);
    expect(stream.starts, 1);

    await stream.stop();
    await stream.stop();
    expect(stream.isActive, isFalse);
    expect(stream.stops, 1);
  });

  test('the hooks do nothing by default', () async {
    final stream = _PushOnlyStream();
    await stream.start();
    expect(stream.isActive, isTrue);
    await stream.stop();
    expect(stream.isActive, isFalse);
  });
}

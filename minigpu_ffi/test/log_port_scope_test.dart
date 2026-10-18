/// The native log port is ONE process-global slot. `setLogCallback(null)` from
/// a platform that holds no registration used to clear it anyway, so any
/// isolate tidying up its own (absent) logger silently cut off the isolate
/// that was actually listening. One that HAD registered cleared the slot even
/// after a newer isolate took it over; it now clears only its own port.
///
/// Log LEVEL stays process-wide by design, so the child below passes the same
/// level as this isolate: only the port is under test.
///
///     dart test test/log_port_scope_test.dart
@TestOn('vm')
library;

import 'dart:async';
import 'dart:isolate';

import 'package:minigpu_ffi/minigpu_ffi.dart';
import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';
import 'package:test/test.dart';

/// mgpu::LogLevel: -1=none 0=debug 1=info 2=warn 3=error.
const int _debug = 0;
const int _none = -1;

/// Child isolate: registers a log callback, hands back a command port, and
/// clears its registration when told to.
Future<void> _registerThenClear(SendPort toMain) async {
  final platform = MinigpuFfi();
  platform.setLogCallback((level, message) {}, level: _debug);
  final commands = ReceivePort();
  toMain.send(commands.sendPort);
  await commands.first;
  platform.setLogCallback(null, level: _debug);
  toMain.send('cleared');
}

void main() {
  late MinigpuFfi platform;

  setUpAll(() async {
    platform = MinigpuFfi();
    await platform.initializeContext();
  });

  tearDownAll(() async {
    platform.setLogCallback(null, level: _none);
    await platform.destroyContext();
  });

  /// Creating a buffer emits native log lines regardless of GPU state.
  void forceNativeLogOutput() {
    platform.createBuffer(256, BufferDataType.float32).destroy();
  }

  test('an isolate that never registered cannot silence this one', () async {
    final lines = <String>[];
    platform.setLogCallback(
      (level, message) => lines.add(message),
      level: _debug,
    );

    forceNativeLogOutput();
    await Future<void>.delayed(const Duration(milliseconds: 200));
    expect(
      lines,
      isNotEmpty,
      reason: 'positive control: native lines must reach this isolate, or '
          'the rest of the test passes vacuously',
    );

    await Isolate.run(() => MinigpuFfi().setLogCallback(null, level: _debug));

    lines.clear();
    forceNativeLogOutput();
    await Future<void>.delayed(const Duration(milliseconds: 200));
    expect(
      lines,
      isNotEmpty,
      reason: 'the child held no registration, so clearing it must not have '
          'dropped this isolate\'s port',
    );
  });

  test('clearing an older registration keeps the newer one', () async {
    final fromChild = ReceivePort();
    addTearDown(fromChild.close);
    final replies = StreamIterator<dynamic>(fromChild);
    await Isolate.spawn(
      _registerThenClear,
      fromChild.sendPort,
      onError: fromChild.sendPort,
    );
    expect(await replies.moveNext(), isTrue);
    final toChild = replies.current as SendPort; // the child is registered

    // Taking the slot over after the child registered.
    final lines = <String>[];
    platform.setLogCallback(
      (level, message) => lines.add(message),
      level: _debug,
    );
    forceNativeLogOutput();
    await Future<void>.delayed(const Duration(milliseconds: 200));
    expect(lines, isNotEmpty, reason: 'positive control');

    toChild.send('clear');
    expect(await replies.moveNext(), isTrue);
    expect(
      replies.current,
      'cleared',
      reason: 'the child failed: ${replies.current}',
    );

    lines.clear();
    forceNativeLogOutput();
    await Future<void>.delayed(const Duration(milliseconds: 200));
    expect(
      lines,
      isNotEmpty,
      reason: 'the child cleared its OWN port, which this isolate had '
          'already replaced, so this isolate must still receive',
    );
  });
}

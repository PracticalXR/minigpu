/// Regression test for the process-global native-log callback lifecycle.
///
/// `mgpuSetLogCallback` is a PROCESS-GLOBAL registry. When Dart registered a
/// [NativeCallable] there, the pointer was owned by ONE isolate — and the VM
/// deletes the trampoline when that isolate exits, while the C++ library keeps
/// the pointer. The next `mgpu::Logger` line, in particular one emitted by a
/// Dawn worker thread (which carries no isolate at all), then aborted the whole
/// VM process:
///
///     runtime_entry.cc: ... Callback invoked after it has been deleted
///     ... isolate_group=(nil)
///
/// A whole-suite `dart test` run reproduces it, because each test FILE is a
/// separate isolate inside ONE VM process — and it also took down unrelated
/// downstream suites (miniav_recorder) that merely link minigpu. This test pins
/// the mechanism on its own: register a log sink here, register one in a
/// short-lived isolate, kill that isolate, then force native log output. Under
/// the old function-pointer path the process dies at that point; under the
/// native-port path the dead port is simply inert.
///
/// It also pins the documented semantics: delivery is PROCESS-GLOBAL and LAST
/// WRITER WINS, so once another isolate registers, this isolate stops receiving
/// until it registers again.
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

/// Entry point for the short-lived isolate. Takes over the process-global
/// registration, reports readiness, then idles until the parent kills it.
///
/// Deliberately does NOT initialise a GPU context: the registration alone is
/// what has to go stale.
void _childRegistersLogSink(SendPort ready) {
  MinigpuFfi().setLogCallback((level, message) {
    // Deliberately empty: what matters is that this isolate OWNS the
    // registration when it dies.
  }, level: _debug);
  ready.send(true);
  // Do not return — the parent kills us, which is what makes the registration
  // go stale mid-process.
}

void main() {
  late MinigpuFfi platform;

  setUpAll(() async {
    platform = MinigpuFfi();
    await platform.initializeContext();
  });

  tearDownAll(() async {
    // Leave the native library on its own stderr sink so later suites in this
    // process are not fed by a port we are about to drop.
    MinigpuFfi().setLogCallback(null, level: _none);
    await platform.destroyContext();
  });

  /// Creating and destroying a buffer emits several `LOG_INFO` lines from
  /// mgpuCreateBuffer regardless of GPU state.
  void forceNativeLogOutput() {
    platform.createBuffer(256, BufferDataType.float32).destroy();
  }

  test('native log survives an isolate that registered and exited', () async {
    final lines = <String>[];
    platform.setLogCallback((level, message) => lines.add(message),
        level: _debug);
    addTearDown(() => platform.setLogCallback(null, level: _none));

    // 1. Positive control for the harness itself: this isolate really does
    //    receive native log lines. Without this the rest could pass vacuously.
    forceNativeLogOutput();
    await Future<void>.delayed(const Duration(milliseconds: 200));
    expect(
      lines,
      isNotEmpty,
      reason: 'no native log lines reached Dart — the test cannot demonstrate '
          'anything about their lifecycle',
    );

    // 2. A second isolate takes over the process-global registration and then
    //    dies, leaving the native side holding its handle.
    final ready = ReceivePort();
    final child = await Isolate.spawn(_childRegistersLogSink, ready.sendPort);
    expect(await ready.first, isTrue);
    ready.close();

    final exited = ReceivePort();
    child.addOnExitListener(exited.sendPort);
    child.kill(priority: Isolate.immediate);
    await exited.first;
    exited.close();

    // 3. THE CRASH POINT. With a NativeCallable the VM aborts the entire
    //    process here. Reaching the next statement is the whole proof.
    lines.clear();
    forceNativeLogOutput();
    await Future<void>.delayed(const Duration(milliseconds: 200));

    // 4. Documented semantics: PROCESS-GLOBAL, LAST WRITER WINS. The dead child
    //    owned the registration, so nothing arrived here — and posting to its
    //    closed port was silent, not fatal.
    expect(
      lines,
      isEmpty,
      reason: 'the last registration (the dead child) should still own the '
          'process-global hook',
    );

    // 5. Re-registering restores delivery to this isolate.
    platform.setLogCallback((level, message) => lines.add(message),
        level: _debug);
    forceNativeLogOutput();
    await Future<void>.delayed(const Duration(milliseconds: 200));
    expect(lines, isNotEmpty);
  });
}

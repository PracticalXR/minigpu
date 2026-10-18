import 'dart:async';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:minigpu/minigpu.dart' show Minigpu;
import 'package:minigpu_platform_interface/minigpu_platform_interface.dart'
    show DeviceBufferLimits, MinigpuPlatform, PlatformSharedOutputTexture;
import 'package:minigpu_view/minigpu_view.dart';

const _channel = MethodChannel('minigpu_view');

/// A healthy device with shared output textures and an always-idle queue.
/// Nothing else may reach it.
class _FakeGpu extends MinigpuPlatform {
  var _textures = 0;

  @override
  Future<void> initializeContext() async {}

  @override
  Future<void> destroyContext() async {}

  @override
  DeviceBufferLimits get currentDeviceBufferLimits => const DeviceBufferLimits(
    deviceGeneration: 1,
    maxBufferSize: 1 << 28,
    maxStorageBufferBindingSize: 1 << 27,
  );

  @override
  Future<void> waitForSubmittedWork() async {}

  @override
  PlatformSharedOutputTexture createSharedOutputTexture(int width, int height) =>
      _FakeTexture(++_textures, width, height);

  @override
  dynamic noSuchMethod(Invocation invocation) =>
      throw StateError('unexpected platform call ${invocation.memberName}');
}

/// Handles differ per texture, as a real texture's do.
class _FakeTexture extends PlatformSharedOutputTexture {
  _FakeTexture(this._id, this.width, this.height);

  final int _id;

  @override
  final int width;

  @override
  final int height;

  @override
  int get d3d11Handle => 0x100 + _id;

  @override
  int get d3d11TexturePtr => 0x200 + _id;

  @override
  void destroy() {}
}

/// Copies at once, or once [landed] completes.
class _Frame extends GpuFrame {
  _Frame(this.width, this.height, [this.landed]);

  @override
  final int width;

  @override
  final int height;

  final Future<void>? landed;

  @override
  Future<bool> copyInto(GpuPaneTarget target) async {
    await landed;
    return true;
  }
}

/// A pane on a fake device, and every host call made for it.
Future<(GpuPane, List<MethodCall>)> _open(WidgetTester tester) async {
  final calls = <MethodCall>[];
  final messenger = tester.binding.defaultBinaryMessenger;
  messenger.setMockMethodCallHandler(_channel, (call) async {
    calls.add(call);
    if (call.method != 'present') return null;
    // One registration per texture, as the Windows host keeps them.
    return {'textureId': (call.arguments as Map)['sharedHandle']};
  });
  addTearDown(() => messenger.setMockMethodCallHandler(_channel, null));
  final gpu = Minigpu.withPlatform(_FakeGpu());
  await gpu.init();
  return (GpuPane(gpu), calls);
}

List<String> _methods(List<MethodCall> calls) => [
  for (final call in calls) call.method,
];

Object? _sharedHandle(MethodCall call) =>
    (call.arguments as Map)['sharedHandle'];

void main() {
  // Frames come only from tester.pump(), so each test decides when the frame
  // showing a new texture has been drawn.
  testWidgets('a resize keeps the old texture until a frame is drawn', (
    tester,
  ) async {
    final (pane, calls) = await _open(tester);
    expect(await pane.showChecked(_Frame(8, 8)), isTrue);
    final old = pane.texture;
    final oldHandle = old.d3d11Handle;
    expect(await pane.showChecked(_Frame(16, 16)), isTrue);
    final current = pane.texture;

    // Nothing cleared or freed between the presents: the view keeps painting
    // the old texture until it has drawn the new one.
    expect(_methods(calls), ['present', 'present']);
    expect(old.isValid, isTrue);
    expect(pane.hasContent, isTrue);
    expect(pane.controller.textureId, current.d3d11Handle);

    await tester.pump();
    expect(_methods(calls), ['present', 'present', 'retire']);
    expect(_sharedHandle(calls.last), oldHandle);
    expect(old.isValid, isFalse);
    expect(current.isValid, isTrue);
    expect(pane.hasContent, isTrue);

    await pane.dispose();
    expect(current.isValid, isFalse);
    expect(calls.last.method, 'dispose');
  });

  testWidgets('a frame retires only the textures it replaced', (tester) async {
    final (pane, calls) = await _open(tester);
    await pane.show(_Frame(8, 8));
    final first = pane.texture;
    await pane.show(_Frame(16, 16));
    final second = pane.texture;
    final secondHandle = second.d3d11Handle;
    // A third size arrives before the frame showing the second is drawn. The
    // second is replaced at once, but stays on screen until the third is shown.
    final landed = Completer<void>();
    final third = pane.showChecked(_Frame(32, 32, landed.future));

    await tester.pump();
    expect(_methods(calls), ['present', 'present', 'retire']);
    expect(first.isValid, isFalse);
    expect(second.isValid, isTrue);

    landed.complete();
    expect(await third, isTrue);
    expect(second.isValid, isTrue);

    await tester.pump();
    expect(_methods(calls), [
      'present',
      'present',
      'retire',
      'present',
      'retire',
    ]);
    expect(_sharedHandle(calls.last), secondHandle);
    expect(second.isValid, isFalse);
    expect(pane.texture.isValid, isTrue);

    await pane.dispose();
  });

  testWidgets('dispose frees a replaced texture without waiting for a frame', (
    tester,
  ) async {
    final (pane, calls) = await _open(tester);
    await pane.show(_Frame(8, 8));
    final old = pane.texture;
    await pane.show(_Frame(16, 16));
    final current = pane.texture;

    await pane.dispose();
    // The controller's own disposal dropped both registrations.
    expect(_methods(calls), ['present', 'present', 'dispose']);
    expect(old.isValid, isFalse);
    expect(current.isValid, isFalse);

    // The retirement that was waiting for a frame finds nothing left.
    await tester.pump();
    expect(_methods(calls), ['present', 'present', 'dispose']);
  });
}

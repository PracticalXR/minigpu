import 'dart:async';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:minigpu_view/minigpu_view.dart';

const _first = RawSharedTexturePreviewSource(
  sharedHandle: 1,
  texturePtr: 2,
  width: 8,
  height: 8,
);
const _second = RawSharedTexturePreviewSource(
  sharedHandle: 3,
  texturePtr: 4,
  width: 16,
  height: 16,
);
const _channel = MethodChannel('minigpu_view');

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  final messenger =
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger;
  tearDown(() => messenger.setMockMethodCallHandler(_channel, null));

  test('retire before any present asks the host nothing', () async {
    final calls = <String>[];
    messenger.setMockMethodCallHandler(_channel, (call) async {
      calls.add(call.method);
      return null;
    });
    final controller = MinigpuPreviewController();
    await controller.retire(_first);
    await controller.dispose();
    expect(calls, isEmpty);
  });

  test('retire names the replaced source and keeps the view', () async {
    final calls = <MethodCall>[];
    messenger.setMockMethodCallHandler(_channel, (call) async {
      calls.add(call);
      return call.method == 'present' ? {'textureId': 7} : null;
    });
    final controller = MinigpuPreviewController();
    await controller.present(_first);
    await controller.present(_second);
    var notifications = 0;
    controller.addListener(() => notifications++);
    await controller.retire(_first);
    expect([for (final c in calls) c.method], ['present', 'present', 'retire']);
    final presented = calls.first.arguments as Map;
    expect(calls.last.arguments, {
      'instanceId': presented['instanceId'],
      'kind': 'nativeSharedTexture',
      'handle': 2,
      'sharedHandle': 1,
      'width': 8,
      'height': 8,
      'pixelFormat': 'rgba8',
    });
    expect(controller.textureId, 7);
    expect(controller.size, const Size(16, 16));
    expect(notifications, 0);
    await controller.dispose();
  });

  test('a host without retire is not an error', () async {
    messenger.setMockMethodCallHandler(_channel, (call) async {
      if (call.method == 'retire') throw MissingPluginException();
      return call.method == 'present' ? {'textureId': 7} : null;
    });
    final controller = MinigpuPreviewController();
    await controller.present(_first);
    await controller.present(_second);
    await expectLater(controller.retire(_first), completes);
    await controller.dispose();
  });

  test('a failed retire is reported', () async {
    messenger.setMockMethodCallHandler(_channel, (call) async {
      if (call.method == 'retire') {
        throw PlatformException(code: 'retire-failed');
      }
      return call.method == 'present' ? {'textureId': 7} : null;
    });
    final controller = MinigpuPreviewController();
    await controller.present(_first);
    await controller.present(_second);
    await expectLater(
      controller.retire(_first),
      throwsA(isA<PlatformException>()),
    );
    await controller.dispose();
  });

  test('retire during clear waits for the host to drop everything', () async {
    final calls = <String>[];
    final dropped = Completer<void>();
    messenger.setMockMethodCallHandler(_channel, (call) async {
      calls.add(call.method);
      if (call.method == 'dispose') await dropped.future;
      return call.method == 'present' ? {'textureId': 7} : null;
    });
    final controller = MinigpuPreviewController();
    await controller.present(_first);
    final clearing = controller.clear();
    var retired = false;
    final retiring = controller.retire(_first).then((_) => retired = true);
    await Future<void>.delayed(Duration.zero);
    expect(retired, isFalse);
    dropped.complete();
    await clearing;
    await retiring;
    expect(calls, ['present', 'dispose']);
    await controller.dispose();
  });

  test('retire after dispose waits for the disposal', () async {
    final calls = <String>[];
    final dropped = Completer<void>();
    messenger.setMockMethodCallHandler(_channel, (call) async {
      calls.add(call.method);
      if (call.method == 'dispose') await dropped.future;
      return call.method == 'present' ? {'textureId': 7} : null;
    });
    final controller = MinigpuPreviewController();
    await controller.present(_first);
    final disposal = controller.dispose();
    var retired = false;
    final retiring = controller.retire(_first).then((_) => retired = true);
    await Future<void>.delayed(Duration.zero);
    expect(retired, isFalse);
    dropped.complete();
    await disposal;
    await retiring;
    expect(calls, ['present', 'dispose']);
  });
}

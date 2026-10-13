import 'dart:async';
import 'package:flutter/widgets.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:minigpu_view/minigpu_view.dart';

const _source = RawSharedTexturePreviewSource(
  sharedHandle: 1,
  texturePtr: 2,
  width: 8,
  height: 8,
);
const _channel = MethodChannel('minigpu_view');

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  final messenger =
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger;
  tearDown(() => messenger.setMockMethodCallHandler(_channel, null));

  test(
    'dispose drains a late registration without publishing its texture id',
    () async {
      final response = Completer<Map<String, Object>>();
      final entered = Completer<void>();
      final calls = <String>[];
      messenger.setMockMethodCallHandler(_channel, (call) async {
        calls.add(call.method);
        if (call.method == 'present') {
          entered.complete();
          return response.future;
        }
        return null;
      });
      final controller = MinigpuPreviewController();
      var notifications = 0;
      controller.addListener(() {
        notifications++;
      });
      final present = controller.present(_source);
      await entered.future;
      var disposed = false;
      final disposal = controller.dispose().then((_) {
        disposed = true;
      });
      await Future<void>.delayed(Duration.zero);
      expect(disposed, isFalse);
      response.complete({'textureId': 7, 'width': 8, 'height': 8});
      await present;
      await disposal;
      expect(calls, ['present', 'dispose']);
      expect(controller.textureId, isNull);
      expect(notifications, 0);
      await expectLater(controller.present(_source), throwsStateError);
    },
  );

  test('clear hides synchronously and can register a fresh surface', () async {
    var id = 6;
    final calls = <String>[];
    messenger.setMockMethodCallHandler(_channel, (call) async {
      calls.add(call.method);
      return call.method == 'present' ? {'textureId': ++id} : null;
    });
    final controller = MinigpuPreviewController();
    await controller.present(_source);
    expect(controller.textureId, 7);
    final clearing = controller.clear();
    expect(controller.textureId, isNull);
    await clearing;
    await controller.present(_source);
    expect(controller.textureId, 8);
    await controller.dispose();
    expect(calls, ['present', 'dispose', 'present', 'dispose']);
  });

  testWidgets('view retirement during build hides before notifying widgets', (
    tester,
  ) async {
    messenger.setMockMethodCallHandler(_channel, (call) async {
      return call.method == 'present' ? {'textureId': 7} : null;
    });
    final controller = MinigpuPreviewController();
    await tester.runAsync(() => controller.present(_source));
    var cleared = false, hiddenDuringBuild = false;
    Future<void>? retirement;
    await tester.pumpWidget(
      Directionality(
        textDirection: TextDirection.ltr,
        child: Column(
          children: [
            AnimatedBuilder(
              animation: controller,
              builder: (_, _) => Text('${controller.textureId}'),
            ),
            Builder(
              builder: (_) {
                if (!cleared) {
                  cleared = true;
                  retirement = controller.clear();
                  hiddenDuringBuild = controller.textureId == null;
                }
                return const SizedBox.shrink();
              },
            ),
          ],
        ),
      ),
    );
    expect(hiddenDuringBuild, isTrue);
    expect(tester.takeException(), isNull);
    await tester.pump();
    expect(find.text('null'), findsOneWidget);
    await tester.runAsync(() => retirement!);
    await tester.pumpWidget(const SizedBox.shrink());
    await tester.runAsync(controller.dispose);
  });

  test(
    'unregister failure is reported instead of claiming safe retirement',
    () async {
      messenger.setMockMethodCallHandler(_channel, (call) async {
        if (call.method == 'present') return {'textureId': 7};
        throw PlatformException(code: 'dispose-failed');
      });
      final controller = MinigpuPreviewController();
      await controller.present(_source);
      await expectLater(
        controller.dispose(),
        throwsA(isA<PlatformException>()),
      );
      expect(controller.textureId, isNull);
    },
  );
}

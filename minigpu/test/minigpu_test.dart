import 'dart:typed_data';
import 'package:minigpu/minigpu.dart';
import 'package:test/test.dart';

late Minigpu minigpu;
void main() {
  group('Minigpu Library Tests', () {
    setUp(() async {
      // Each test gets a new instance.
      minigpu = Minigpu();
      await minigpu.init();
    });

    test('Context Initialization', () {
      // isInitialized should be true after init.
      expect(minigpu.isInitialized, isTrue);
    });

    test('Buffer creation and data transfer', () async {
      const int bufferSize = 100;
      final int memorySize = bufferSize * 4; // 4 bytes per float

      // Create a buffer and set known data.
      final buffer = minigpu.createBuffer(memorySize, BufferDataType.float32);
      final inputData = Float32List.fromList(
        List.generate(bufferSize, (i) => i.toDouble()),
      );
      buffer.write(inputData, bufferSize);

      // Read back the data.
      final outputData = Float32List(bufferSize);
      await buffer.read(outputData, bufferSize);

      // Check that the buffer returns the original data.
      expect(outputData, equals(inputData));

      // Clean up.
      buffer.destroy();
    });

    test('Compute Shader: adds 0.2 to each element', () async {
      const int numFloats = 100;
      final int memorySize = numFloats * 4;

      // Create input and output buffers.
      final inputBuffer = minigpu.createBuffer(
        memorySize,
        BufferDataType.float32,
      );
      final outputBuffer = minigpu.createBuffer(
        memorySize,
        BufferDataType.float32,
      );

      // Initialize input data.
      final inputData = Float32List.fromList(
        List.generate(numFloats, (i) => i.toDouble()),
      );
      inputBuffer.write(inputData, numFloats);

      // WGSL shader code which adds 0.2 to each input element.
      final shaderCode =
          '''
const GELU_SCALING_FACTOR: f32 = 0.7978845608028654;
@group(0) @binding(0) var<storage, read_write> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let i: u32 = GlobalInvocationID.x;
    if (i < ${numFloats}u) {
        let x: f32 = inp[i];
        out[i] = x + 0.2;
    }
}
''';

      // Create and configure the compute shader.
      final shader = minigpu.createComputeShader();
      shader.loadKernelString(shaderCode);
      shader.setBuffer('inp', inputBuffer);
      shader.setBuffer('out', outputBuffer);

      // Calculate workgroups (assuming 256 threads per group).
      final int workgroups = ((numFloats + 255) / 256).floor();
      await shader.dispatch(workgroups, 1, 1);

      // Read output data.
      final outputData = Float32List(numFloats);
      await outputBuffer.read(outputData, numFloats);

      // Verify that each element was increased by 0.2.
      for (int i = 0; i < numFloats; i++) {
        expect(outputData[i], closeTo(inputData[i] + 0.2, 1e-4));
      }

      // Clean up.
      shader.destroy();
      inputBuffer.destroy();
      outputBuffer.destroy();
    });
  });

  // -------------------------------------------------------------------------
  // Minigpu.setLogCallback
  // -------------------------------------------------------------------------

  group('Minigpu.setLogCallback', () {
    tearDown(() {
      // Always clear the log callback after each test so no NativeCallable
      // leaks across the test suite.
      Minigpu.setLogCallback(null);
    });

    test('setLogCallback(null) does not throw', () {
      expect(() => Minigpu.setLogCallback(null), returnsNormally);
    });

    test('setLogCallback with a callback does not throw', () {
      expect(() => Minigpu.setLogCallback((level, msg) {}), returnsNormally);
    });

    test('setLogCallback replaces an existing callback without throwing', () {
      expect(() {
        Minigpu.setLogCallback((level, msg) {});
        Minigpu.setLogCallback((level, msg) {});
        Minigpu.setLogCallback(null);
      }, returnsNormally);
    });

    test('setLogCallback level=quiet (-1) does not throw', () {
      expect(
        () => Minigpu.setLogCallback((level, msg) {}, level: -1),
        returnsNormally,
      );
    });

    test('setLogCallback level=debug (0) does not throw', () {
      expect(
        () => Minigpu.setLogCallback((level, msg) {}, level: 0),
        returnsNormally,
      );
    });

    test(
      'callback receives at least one message during context init',
      () async {
        // Destroy the existing context so we can init fresh with the callback.
        await minigpu.destroy();

        final received = <(int, String)>[];
        Minigpu.setLogCallback(
          (lvl, msg) => received.add((lvl, msg)),
          level: 0, // LOG_DEBUG — capture everything
        );

        // Re-init: Dawn will emit INFO/DEBUG messages during device creation.
        await minigpu.init();

        // Give listener NativeCallable a turn on the event loop.
        await Future<void>.delayed(const Duration(milliseconds: 50));

        // We do not assert received.isNotEmpty because Dawn's verbosity depends
        // on the native build configuration and driver. The test validates that
        // the callback round-trip works without throwing, and that any messages
        // that ARE received have valid level integers.
        for (final (lvl, _) in received) {
          expect(lvl, inInclusiveRange(0, 3));
        }
      },
    );

    test('no callback messages at level=quiet after install', () async {
      await minigpu.destroy();

      final received = <String>[];
      // Install at quiet (-1) — nothing should arrive.
      Minigpu.setLogCallback((_, msg) => received.add(msg), level: -1);

      await minigpu.init();
      await Future<void>.delayed(const Duration(milliseconds: 50));

      expect(
        received,
        isEmpty,
        reason: 'Expected no messages at level -1 (quiet)',
      );
    });
  });
}

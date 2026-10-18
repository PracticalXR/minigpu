import 'package:test/test.dart';
import 'package:gpu_tensor/gpu_tensor.dart';
import 'package:gpu_pipeline/src/pipeline_ports.dart';

void main() {
  group('ValidationResult', () {
    test('creates valid result', () {
      final result = ValidationResult.valid();
      expect(result.isValid, isTrue);
      expect(result.errors, isEmpty);
      expect(result.warnings, isEmpty);
    });

    test('creates valid result with warnings', () {
      final result = ValidationResult.valid(
        warnings: ['Warning 1', 'Warning 2'],
      );
      expect(result.isValid, isTrue);
      expect(result.warnings, hasLength(2));
      expect(result.warnings.first, equals('Warning 1'));
    });

    test('creates invalid result', () {
      final result = ValidationResult.invalid('Test error');
      expect(result.isValid, isFalse);
      expect(result.errors, hasLength(1));
      expect(result.errors.first, equals('Test error'));
    });

    test('creates invalid result with multiple errors', () {
      final result = ValidationResult.invalidMultiple(['Error 1', 'Error 2']);
      expect(result.isValid, isFalse);
      expect(result.errors, hasLength(2));
    });
  });

  group('FlexibleInputPort', () {
    test('accepts any tensor when no constraints', () async {
      final port = FlexibleInputPort(name: 'test');

      final tensor1D = await Tensor.create([100]);
      final tensor2D = await Tensor.create([10, 20]);
      final tensor3D = await Tensor.create([5, 10, 3]);

      expect(port.validateInput(tensor1D).isValid, isTrue);
      expect(port.validateInput(tensor2D).isValid, isTrue);
      expect(port.validateInput(tensor3D).isValid, isTrue);
    });

    test('enforces minimum dimensions', () async {
      final port = FlexibleInputPort(name: 'test', minDimensions: 2);

      final tensor1D = await Tensor.create([100]);
      final tensor2D = await Tensor.create([10, 20]);

      expect(port.validateInput(tensor1D).isValid, isFalse);
      expect(port.validateInput(tensor2D).isValid, isTrue);
    });

    test('enforces maximum dimensions', () async {
      final port = FlexibleInputPort(name: 'test', maxDimensions: 2);

      final tensor2D = await Tensor.create([10, 20]);
      final tensor3D = await Tensor.create([5, 10, 3]);

      expect(port.validateInput(tensor2D).isValid, isTrue);
      expect(port.validateInput(tensor3D).isValid, isFalse);
    });

    test('handles null input for required port', () {
      final port = FlexibleInputPort(name: 'test', isRequired: true);
      expect(port.validateInput(null).isValid, isFalse);
    });

    test('handles null input for optional port', () {
      final port = FlexibleInputPort(name: 'test', isRequired: false);
      expect(port.validateInput(null).isValid, isTrue);
    });
  });

  group('AudioInputPort', () {
    test('validates mono audio (1D tensor)', () async {
      final port = AudioInputPort(name: 'audio');
      final monoAudio = await Tensor.create([1024]);

      expect(port.validateInput(monoAudio).isValid, isTrue);
    });

    test('validates stereo audio (2D tensor)', () async {
      final port = AudioInputPort(name: 'audio');
      final stereoAudio = await Tensor.create([
        2,
        1024,
      ]); // 2 channels, 1024 samples

      expect(port.validateInput(stereoAudio).isValid, isTrue);
    });

    test('rejects tensors with too many dimensions', () async {
      final port = AudioInputPort(name: 'audio');
      final invalidAudio = await Tensor.create([2, 1024, 3]); // 3D tensor

      expect(port.validateInput(invalidAudio).isValid, isFalse);
    });

    test('enforces minimum sample count', () async {
      final port = AudioInputPort(name: 'audio', minSamples: 512);

      final shortAudio = await Tensor.create([256]);
      final longAudio = await Tensor.create([1024]);

      expect(port.validateInput(shortAudio).isValid, isFalse);
      expect(port.validateInput(longAudio).isValid, isTrue);
    });

    test('enforces maximum sample count', () async {
      final port = AudioInputPort(name: 'audio', maxSamples: 2048);

      final normalAudio = await Tensor.create([1024]);
      final longAudio = await Tensor.create([4096]);

      expect(port.validateInput(normalAudio).isValid, isTrue);
      expect(port.validateInput(longAudio).isValid, isFalse);
    });

    test('validates expected channel count', () async {
      final port = AudioInputPort(name: 'audio', expectedChannels: 2);

      final monoAudio = await Tensor.create([1024]); // 1 channel
      final stereoAudio = await Tensor.create([2, 1024]); // 2 channels
      final surroundAudio = await Tensor.create([6, 1024]); // 6 channels

      expect(port.validateInput(monoAudio).isValid, isFalse);
      expect(port.validateInput(stereoAudio).isValid, isTrue);
      expect(port.validateInput(surroundAudio).isValid, isFalse);
    });
  });

  group('VideoInputPort', () {
    test('validates 3D video tensor [H,W,C]', () async {
      final port = VideoInputPort(name: 'video');
      final video = await Tensor.create([720, 1280, 3]); // HD video, RGB

      expect(port.validateInput(video).isValid, isTrue);
    });

    test('validates 4D batched video tensor [B,H,W,C]', () async {
      final port = VideoInputPort(name: 'video');
      final batchedVideo = await Tensor.create([
        4,
        720,
        1280,
        3,
      ]); // Batch of 4 HD frames

      expect(port.validateInput(batchedVideo).isValid, isTrue);
    });

    test('rejects invalid dimensions', () async {
      final port = VideoInputPort(name: 'video');

      final tensor1D = await Tensor.create([1000]);
      final tensor2D = await Tensor.create([720, 1280]);
      final tensor5D = await Tensor.create([4, 720, 1280, 3, 2]);

      expect(port.validateInput(tensor1D).isValid, isFalse);
      expect(port.validateInput(tensor2D).isValid, isFalse);
      expect(port.validateInput(tensor5D).isValid, isFalse);
    });

    test('enforces width constraints', () async {
      final port = VideoInputPort(name: 'video', minWidth: 640, maxWidth: 1920);

      final smallVideo = await Tensor.create([
        480,
        320,
        3,
      ]); // 320px wide (too small)
      final normalVideo = await Tensor.create([
        720,
        1280,
        3,
      ]); // 1280px wide (good)
      final largeVideo = await Tensor.create([
        1080,
        3840,
        3,
      ]); // 3840px wide (too large)

      expect(port.validateInput(smallVideo).isValid, isFalse);
      expect(port.validateInput(normalVideo).isValid, isTrue);
      expect(port.validateInput(largeVideo).isValid, isFalse);
    });

    test('enforces height constraints', () async {
      final port = VideoInputPort(
        name: 'video',
        minHeight: 480,
        maxHeight: 1080,
      );

      final smallVideo = await Tensor.create([
        240,
        1280,
        3,
      ]); // 240px tall (too small)
      final normalVideo = await Tensor.create([
        720,
        1280,
        3,
      ]); // 720px tall (good)
      final largeVideo = await Tensor.create([
        2160,
        1280,
        3,
      ]); // 2160px tall (too large)

      expect(port.validateInput(smallVideo).isValid, isFalse);
      expect(port.validateInput(normalVideo).isValid, isTrue);
      expect(port.validateInput(largeVideo).isValid, isFalse);
    });
  });

  group('ExactInputPort', () {
    test('accepts tensor with exact matching shape', () async {
      final port = ExactInputPort(name: 'exact', expectedShape: [224, 224, 3]);
      final tensor = await Tensor.create([224, 224, 3]);

      expect(port.validateInput(tensor).isValid, isTrue);
    });

    test('rejects tensor with wrong shape', () async {
      final port = ExactInputPort(name: 'exact', expectedShape: [224, 224, 3]);

      final wrongSize = await Tensor.create([256, 256, 3]);
      final wrongChannels = await Tensor.create([224, 224, 4]);
      final wrongDimensions = await Tensor.create([224, 224]);

      expect(port.validateInput(wrongSize).isValid, isFalse);
      expect(port.validateInput(wrongChannels).isValid, isFalse);
      expect(port.validateInput(wrongDimensions).isValid, isFalse);
    });

    test('provides expected shape', () {
      final port = ExactInputPort(name: 'exact', expectedShape: [100, 200, 3]);
      expect(port.expectedShape, equals([100, 200, 3]));
    });
  });

  group('CustomInputPort', () {
    test('uses custom validator function', () async {
      bool validatorCalled = false;
      final port = CustomInputPort(
        name: 'custom',
        validator: (tensor) {
          validatorCalled = true;
          if (tensor == null) return ValidationResult.invalid('Null tensor');
          if (tensor.shape.first > 1000) {
            return ValidationResult.invalid('Too large');
          }
          return ValidationResult.valid();
        },
      );

      final validTensor = await Tensor.create([500]);
      final invalidTensor = await Tensor.create([2000]);

      expect(port.validateInput(validTensor).isValid, isTrue);
      expect(port.validateInput(invalidTensor).isValid, isFalse);
      expect(validatorCalled, isTrue);
    });

    test('handles null input for optional port', () async {
      final port = CustomInputPort(
        name: 'custom',
        validator: (tensor) {
          return tensor == null
              ? ValidationResult.invalid('Should not be null')
              : ValidationResult.valid();
        },
      );

      expect(port.validateInput(null).isValid, isFalse);
    });
  });

  group('StaticInputPort', () {
    test('validates loaded asset', () async {
      final port = StaticInputPort(
        name: 'weights',
        assetPath: 'assets/model.bin',
      );
      final tensor = await Tensor.create([1000, 512]);

      expect(port.validateInput(tensor).isValid, isTrue);
    });

    test('rejects null asset', () {
      final port = StaticInputPort(
        name: 'weights',
        assetPath: 'assets/model.bin',
      );
      expect(port.validateInput(null).isValid, isFalse);
    });

    test('is always persistent', () {
      final port = StaticInputPort(
        name: 'weights',
        assetPath: 'assets/model.bin',
      );
      expect(port.persistent, isTrue);
    });
  });

  group('FixedOutputPort', () {
    test('returns fixed shape regardless of inputs', () async {
      final port = FixedOutputPort(name: 'output', shape: [100, 200]);

      final inputs = {
        'input': await Tensor.create([50, 75]),
      };
      final parameters = <String, dynamic>{};

      expect(port.calculateOutputShape(inputs, parameters), equals([100, 200]));
    });
  });

  group('DynamicOutputPort', () {
    test('uses fixed shape when provided', () async {
      final port = DynamicOutputPort(name: 'output', fixedShape: [300, 400]);

      final inputs = {
        'input': await Tensor.create([100, 100]),
      };
      final parameters = <String, dynamic>{};

      expect(port.calculateOutputShape(inputs, parameters), equals([300, 400]));
    });

    test('uses shape calculator when provided', () async {
      final port = DynamicOutputPort(
        name: 'output',
        shapeCalculator: (inputs, params) {
          final inputShape = inputs['input']!.shape;
          final scale = params['scale'] as int? ?? 1;
          return [inputShape[0] * scale, inputShape[1] * scale];
        },
      );

      final inputs = {
        'input': await Tensor.create([100, 200]),
      };
      final parameters = {'scale': 2};

      expect(port.calculateOutputShape(inputs, parameters), equals([200, 400]));
    });

    test('falls back to first input shape', () async {
      final port = DynamicOutputPort(name: 'output');

      final inputs = {
        'input': await Tensor.create([50, 75, 3]),
      };
      final parameters = <String, dynamic>{};

      expect(
        port.calculateOutputShape(inputs, parameters),
        equals([50, 75, 3]),
      );
    });

    test('throws when no shape can be determined', () {
      final port = DynamicOutputPort(name: 'output');

      final inputs = <String, Tensor>{};
      final parameters = <String, dynamic>{};

      expect(
        () => port.calculateOutputShape(inputs, parameters),
        throwsException,
      );
    });
  });

  group('TextureOutputPort', () {
    test('calculates texture output shape correctly', () {
      final port = TextureOutputPort(
        name: 'texture',
        width: 800,
        height: 600,
        channels: 4,
      );

      final inputs = <String, Tensor>{};
      final parameters = <String, dynamic>{};

      final expectedSize = 800 * 600 * 4; // width * height * channels
      expect(
        port.calculateOutputShape(inputs, parameters),
        equals([expectedSize]),
      );
    });

    test('provides texture dimensions', () {
      final port = TextureOutputPort(
        name: 'texture',
        width: 1920,
        height: 1080,
        channels: 3,
      );

      expect(port.textureDimensions, equals([1920, 1080, 3]));
      expect(port.pixelCount, equals(1920 * 1080));
      expect(port.width, equals(1920));
      expect(port.height, equals(1080));
      expect(port.channels, equals(3));
    });

    test('defaults to RGBA (4 channels)', () {
      final port = TextureOutputPort(name: 'texture', width: 256, height: 256);
      expect(port.channels, equals(4));
      expect(port.format, equals('rgba_texture'));
    });
  });

  group('HistoryOutputPort', () {
    test('adds time dimension to input shape', () async {
      final port = HistoryOutputPort(
        name: 'history',
        timeSlices: 10,
        description: "History of inputs",
      );

      final inputs = {
        'input': await Tensor.create([100, 200]),
      };
      final parameters = <String, dynamic>{};

      expect(
        port.calculateOutputShape(inputs, parameters),
        equals([10, 100, 200]),
      );
    });

    test('handles empty inputs', () {
      final port = HistoryOutputPort(
        name: 'history',
        timeSlices: 5,
        description: "History of inputs",
      );

      final inputs = <String, Tensor>{};
      final parameters = <String, dynamic>{};

      expect(port.calculateOutputShape(inputs, parameters), equals([5]));
    });

    test('uses history buffer format', () {
      final port = HistoryOutputPort(
        name: 'history',
        timeSlices: 8,
        description: "History buffer",
      );
      expect(port.format, equals('history_buffer'));
    });
  });

  group('InputPortCollection', () {
    test('adds and retrieves ports', () {
      final collection = InputPortCollection();
      final port1 = FlexibleInputPort(name: 'input1');
      final port2 = AudioInputPort(name: 'audio');

      collection.add(port1);
      collection.add(port2);

      expect(collection['input1'], equals(port1));
      expect(collection['audio'], equals(port2));
      expect(collection.length, equals(2));
      expect(collection.names, containsAll(['input1', 'audio']));
    });

    test('identifies required and optional ports', () {
      final collection = InputPortCollection();
      collection.add(FlexibleInputPort(name: 'required', isRequired: true));
      collection.add(FlexibleInputPort(name: 'optional', isRequired: false));

      expect(collection.required, hasLength(1));
      expect(collection.required.first.name, equals('required'));
    });

    test('validates all inputs successfully', () async {
      final collection = InputPortCollection();
      collection.add(FlexibleInputPort(name: 'input1', isRequired: true));
      collection.add(FlexibleInputPort(name: 'input2', isRequired: false));

      final inputs = {
        'input1': await Tensor.create([100]),
        'input2': await Tensor.create([200]),
      };

      final result = collection.validateAll(inputs);
      expect(result.isValid, isTrue);
    });

    test('validates with missing required input', () {
      final collection = InputPortCollection();
      collection.add(FlexibleInputPort(name: 'required', isRequired: true));
      collection.add(FlexibleInputPort(name: 'optional', isRequired: false));

      final inputs = <String, Tensor>{}; // Missing required input

      final result = collection.validateAll(inputs);
      expect(result.isValid, isFalse);
      expect(result.errors.first, contains('required'));
    });

    test('validates with missing optional input', () async {
      final collection = InputPortCollection();
      collection.add(FlexibleInputPort(name: 'required', isRequired: true));
      collection.add(FlexibleInputPort(name: 'optional', isRequired: false));

      final inputs = {
        'required': await Tensor.create([100]),
        // optional input missing - should be OK
      };

      final result = collection.validateAll(inputs);
      expect(result.isValid, isTrue);
    });
  });

  group('OutputPortCollection', () {
    test('adds and retrieves ports', () {
      final collection = OutputPortCollection();
      final port1 = FixedOutputPort(name: 'output1', shape: [100]);
      final port2 = TextureOutputPort(name: 'texture', width: 800, height: 600);

      collection.add(port1);
      collection.add(port2);

      expect(collection['output1'], equals(port1));
      expect(collection['texture'], equals(port2));
      expect(collection.length, equals(2));
    });

    test('creates output tensors', () async {
      final collection = OutputPortCollection();
      collection.add(FixedOutputPort(name: 'fixed', shape: [50, 100]));
      collection.add(
        TextureOutputPort(name: 'texture', width: 10, height: 10, channels: 4),
      );

      final inputs = <String, Tensor>{};
      final parameters = <String, dynamic>{};

      final outputs = await collection.createOutputTensors(inputs, parameters);

      expect(outputs, hasLength(2));
      expect(outputs['fixed']?.shape, equals([50, 100]));
      expect(outputs['texture']?.shape, equals([400])); // 10*10*4
    });
  });

  group('GeneratedInputPort', () {
    test('creates port with generation parameters', () {
      final port = GeneratedInputPort(
        name: 'noise',
        shape: [1024],
        generatorType: 'white_noise',
        generatorParameters: {'amplitude': 1.0, 'seed': 42},
      );

      expect(port.expectedShape, equals([1024]));
      expect(port.generatorType, equals('white_noise'));
      expect(port.generatorParameters['amplitude'], equals(1.0));
      expect(port.format, equals('generated'));
    });

    test('always validates as valid', () {
      final port = GeneratedInputPort(
        name: 'pattern',
        shape: [100, 100],
        generatorType: 'checkerboard',
      );

      final result = port.validateInput(null);
      expect(result.isValid, isTrue);
    });
  });

  group('ParameterOutputPort', () {
    test('tracks parameter dependencies', () {
      final port = ParameterOutputPort(
        name: 'adaptive',
        parameterDependencies: ['width', 'height', 'channels'],
        shapeCalculator: (inputs, params) {
          final w = params['width'] as int;
          final h = params['height'] as int;
          final c = params['channels'] as int;
          return [w * h * c];
        },
      );

      expect(port.parameterDependencies, hasLength(3));
      expect(
        port.parameterDependencies,
        containsAll(['width', 'height', 'channels']),
      );
    });

    test('validates parameter presence', () {
      final port = ParameterOutputPort(
        name: 'adaptive',
        parameterDependencies: ['width', 'height'],
        shapeCalculator: (inputs, params) => [100],
      );

      final validParams = {'width': 800, 'height': 600};
      final invalidParams = {'width': 800}; // missing height

      expect(port.validateParameters(validParams).isValid, isTrue);
      expect(port.validateParameters(invalidParams).isValid, isFalse);
    });

    test('calculates shape using parameters', () {
      final port = ParameterOutputPort(
        name: 'adaptive',
        parameterDependencies: ['scale'],
        shapeCalculator: (inputs, params) {
          final scale = params['scale'] as int;
          return [100 * scale, 200 * scale];
        },
      );

      final inputs = <String, Tensor>{};
      final parameters = {'scale': 3};

      expect(port.calculateOutputShape(inputs, parameters), equals([300, 600]));
    });
  });
}

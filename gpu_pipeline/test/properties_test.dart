import 'package:test/test.dart';
import 'package:gpu_pipeline/src/pipeline_properties.dart';

enum TestColorScheme { red, green, blue, rainbow, plasma }

enum TestFilterType { lowpass, highpass, bandpass, notch }

void main() {
  group('EnumRegistry', () {
    setUp(() {
      // Clear registry before each test
      EnumRegistry.clear();
    });

    test('registers enum types correctly', () {
      EnumRegistry.register(TestColorScheme.values);

      expect(EnumRegistry.isRegistered('TestColorScheme'), isTrue);
      expect(EnumRegistry.getValues('TestColorScheme'), hasLength(5));
    });

    test('retrieves enum values by name', () {
      EnumRegistry.register(TestColorScheme.values);

      final redValue = EnumRegistry.getValueByName(
        'TestColorScheme',
        'TestColorScheme.red',
      );
      expect(redValue, equals(TestColorScheme.red));

      final invalidValue = EnumRegistry.getValueByName(
        'TestColorScheme',
        'invalid',
      );
      expect(invalidValue, isNull);
    });

    test('retrieves enum values by index', () {
      EnumRegistry.register(TestColorScheme.values);

      final firstValue = EnumRegistry.getValueByIndex('TestColorScheme', 0);
      expect(firstValue, equals(TestColorScheme.red));

      final outOfBounds = EnumRegistry.getValueByIndex('TestColorScheme', 10);
      expect(outOfBounds, isNull);
    });

    test('handles multiple enum types', () {
      EnumRegistry.register(TestColorScheme.values);
      EnumRegistry.register(TestFilterType.values);

      expect(EnumRegistry.registeredTypes, hasLength(2));
      expect(
        EnumRegistry.registeredTypes,
        containsAll(['TestColorScheme', 'TestFilterType']),
      );
    });

    test('handles empty enum list gracefully', () {
      EnumRegistry.register(<TestColorScheme>[]);
      expect(EnumRegistry.isRegistered('TestColorScheme'), isFalse);
    });
  });

  group('Parameter<T>', () {
    group('validation', () {
      test('validates correct types', () {
        final intParam = Parameter.stageInt(
          'test_int',
          defaultValue: 5,
          description: 'Test integer parameter',
        );

        expect(intParam.validate(10).isValid, isTrue);
        expect(intParam.validate(5.5).isValid, isFalse);
        expect(intParam.validate('string').isValid, isFalse);
      });

      test('validates numeric ranges', () {
        final rangedParam = Parameter.stageFloat(
          'gain',
          defaultValue: 1.0,
          min: 0.0,
          max: 10.0,
          description: 'Audio gain parameter',
        );

        expect(rangedParam.validate(5.0).isValid, isTrue);
        expect(rangedParam.validate(0.0).isValid, isTrue);
        expect(rangedParam.validate(10.0).isValid, isTrue);
        expect(rangedParam.validate(-1.0).isValid, isFalse);
        expect(rangedParam.validate(11.0).isValid, isFalse);
      });

      test('validates allowed values', () {
        final restrictedParam = Parameter.stageString(
          'mode',
          defaultValue: 'auto',
          allowedValues: ['auto', 'manual', 'custom'],
          description: 'Processing mode',
        );

        expect(restrictedParam.validate('auto').isValid, isTrue);
        expect(restrictedParam.validate('manual').isValid, isTrue);
        expect(restrictedParam.validate('invalid').isValid, isFalse);
      });

      test('validates enum parameters', () {
        final enumParam = Parameter.stageEnum(
          'color_scheme',
          defaultValue: TestColorScheme.red,
          allValues: TestColorScheme.values,
          description: 'Color scheme selection',
        );

        expect(enumParam.validate(TestColorScheme.blue).isValid, isTrue);
        expect(enumParam.validate('invalid').isValid, isFalse);
        expect(enumParam.validate(42).isValid, isFalse);
      });

      test('validates vector parameters', () {
        final vec3Param = Parameter.stageVec3(
          'position',
          defaultValue: [0.0, 0.0, 0.0],
          description: '3D position vector',
        );

        expect(vec3Param.validate([1.0, 2.0, 3.0]).isValid, isTrue);
        expect(vec3Param.validate('not a list').isValid, isFalse);
        expect(vec3Param.validate(123).isValid, isFalse);
      });
    });

    group('sanitization', () {
      test('clamps numeric values to valid range', () {
        final rangedParam = Parameter.stageFloat(
          'volume',
          defaultValue: 0.5,
          min: 0.0,
          max: 1.0,
          description: 'Volume level',
        );

        expect(rangedParam.sanitize(-0.5), equals(0.0));
        expect(rangedParam.sanitize(1.5), equals(1.0));
        expect(rangedParam.sanitize(0.7), equals(0.7));
      });

      test('falls back to default for invalid types', () {
        final intParam = Parameter.stageInt(
          'count',
          defaultValue: 10,
          description: 'Item count',
        );

        expect(intParam.sanitize('invalid'), equals(10));
        expect(intParam.sanitize(5.5), equals(10));
        expect(intParam.sanitize(15), equals(15));
      });

      test('uses default when value not in allowed list', () {
        final restrictedParam = Parameter.stageString(
          'quality',
          defaultValue: 'medium',
          allowedValues: ['low', 'medium', 'high'],
          description: 'Quality setting',
        );

        expect(restrictedParam.sanitize('invalid'), equals('medium'));
        expect(restrictedParam.sanitize('high'), equals('high'));
      });
    });

    group('serialization', () {
      test('serializes basic types correctly', () {
        final boolParam = Parameter.stageBool(
          'enabled',
          defaultValue: true,
          description: 'Enable feature',
        );
        final intParam = Parameter.stageInt(
          'count',
          defaultValue: 42,
          description: 'Count value',
        );
        final floatParam = Parameter.stageFloat(
          'ratio',
          defaultValue: 3.14,
          description: 'Ratio value',
        );
        final stringParam = Parameter.stageString(
          'name',
          defaultValue: 'test',
          description: 'Name value',
        );

        expect(boolParam.serialize(true), equals('true'));
        expect(intParam.serialize(42), equals('42'));
        expect(floatParam.serialize(3.14), equals('3.14'));
        expect(stringParam.serialize('hello'), equals('hello'));
      });

      test('serializes enum values correctly', () {
        final enumParam = Parameter.stageEnum(
          'scheme',
          defaultValue: TestColorScheme.red,
          allValues: TestColorScheme.values,
          description: 'Color scheme',
        );

        final serialized = enumParam.serialize(TestColorScheme.blue);
        expect(serialized, contains('TestColorScheme'));
        expect(serialized, contains('blue'));
      });

      test('serializes list values correctly', () {
        final vectorParam = Parameter.stageVec3(
          'position',
          defaultValue: [1.0, 2.0, 3.0],
          description: 'Position vector',
        );

        expect(vectorParam.serialize([4.0, 5.0, 6.0]), equals('4.0,5.0,6.0'));
      });
    });

    group('deserialization', () {
      test('deserializes basic types correctly', () {
        final boolParam = Parameter.stageBool(
          'flag',
          defaultValue: false,
          description: 'Boolean flag',
        );
        final intParam = Parameter.stageInt(
          'num',
          defaultValue: 0,
          description: 'Integer value',
        );
        final floatParam = Parameter.stageFloat(
          'val',
          defaultValue: 0.0,
          description: 'Float value',
        );
        final stringParam = Parameter.stageString(
          'text',
          defaultValue: '',
          description: 'Text value',
        );

        expect(boolParam.deserialize('true'), isTrue);
        expect(boolParam.deserialize('false'), isFalse);
        expect(intParam.deserialize('123'), equals(123));
        expect(floatParam.deserialize('4.56'), equals(4.56));
        expect(stringParam.deserialize('hello'), equals('hello'));
      });

      test('falls back to default on invalid data', () {
        final intParam = Parameter.stageInt(
          'num',
          defaultValue: 42,
          description: 'Integer value',
        );

        expect(intParam.deserialize('invalid'), equals(42));
        expect(intParam.deserialize(''), equals(42));
      });
    });

    group('factory constructors', () {
      test('creates float parameters correctly', () {
        final param = Parameter.stageFloat(
          'gain',
          defaultValue: 1.0,
          min: 0.0,
          max: 2.0,
          description: 'Audio gain',
        );

        expect(param.name, equals('gain'));
        expect(param.defaultValue, equals(1.0));
        expect(param.min, equals(0.0));
        expect(param.max, equals(2.0));
        expect(param.description, equals('Audio gain'));
      });

      test('creates vector parameters with correct dimensions', () {
        final vec2 = Parameter.stageVec2(
          'pos2d',
          defaultValue: [1.0, 2.0],
          description: '2D position',
        );
        final vec3 = Parameter.stageVec3(
          'pos3d',
          defaultValue: [1.0, 2.0, 3.0],
          description: '3D position',
        );
        final vec4 = Parameter.stageVec4(
          'color',
          defaultValue: [1.0, 0.5, 0.0, 1.0],
          description: 'RGBA color',
        );

        expect(vec2.defaultValue, hasLength(2));
        expect(vec3.defaultValue, hasLength(3));
        expect(vec4.defaultValue, hasLength(4));
      });

      test('validates vector dimensions in constructors', () {
        expect(
          () => Parameter.stageVec2(
            'bad',
            defaultValue: [1.0],
            description: 'Bad vec2',
          ),
          throwsA(isA<AssertionError>()),
        );

        expect(
          () => Parameter.stageVec3(
            'bad',
            defaultValue: [1.0, 2.0],
            description: 'Bad vec3',
          ),
          throwsA(isA<AssertionError>()),
        );
      });

      test('automatically registers enum types', () {
        Parameter.stageEnum(
          'filter',
          defaultValue: TestFilterType.lowpass,
          allValues: TestFilterType.values,
          description: 'Filter type',
        );

        expect(EnumRegistry.isRegistered('TestFilterType'), isTrue);
        expect(EnumRegistry.getValues('TestFilterType'), hasLength(4));
      });
    });
  });

  group('StageProperties mixin', () {
    late TestStage stage;

    setUp(() {
      stage = TestStage();
    });

    test('adds and retrieves parameters', () {
      final param = Parameter.stageFloat(
        'gain',
        defaultValue: 1.0,
        description: 'Audio gain',
      );
      stage.addParameter(param);

      expect(stage.hasParameter('gain'), isTrue);
      expect(stage.getParameter<double>('gain'), equals(1.0));
      expect(stage.parameterDefinitions, hasLength(1));
    });

    test('updates parameters with validation', () async {
      final param = Parameter.stageFloat(
        'volume',
        defaultValue: 0.5,
        min: 0.0,
        max: 1.0,
        description: 'Volume level',
      );
      stage.addParameter(param);

      await stage.updateParameter('volume', 0.8);
      expect(stage.getParameter<double>('volume'), equals(0.8));

      // Should throw on invalid value
      expect(
        () async => await stage.updateParameter('volume', 2.0),
        throwsArgumentError,
      );
    });

    test('updates multiple parameters atomically', () async {
      stage.addParameter(
        Parameter.stageFloat('gain', defaultValue: 1.0, description: 'Gain'),
      );
      stage.addParameter(
        Parameter.stageInt('delay', defaultValue: 100, description: 'Delay'),
      );
      stage.addParameter(
        Parameter.stageBool(
          'enabled',
          defaultValue: true,
          description: 'Enabled',
        ),
      );

      await stage.updateParameters({
        'gain': 1.5,
        'delay': 200,
        'enabled': false,
      });

      expect(stage.getParameter<double>('gain'), equals(1.5));
      expect(stage.getParameter<int>('delay'), equals(200));
      expect(stage.getParameter<bool>('enabled'), isFalse);
    });

    test('validates parameter existence', () {
      expect(stage.hasParameter('nonexistent'), isFalse);
      expect(
        () => stage.getParameter<double>('nonexistent'),
        throwsArgumentError,
      );
    });

    test('provides type-safe parameter access', () {
      stage.addParameter(
        Parameter.stageInt('count', defaultValue: 10, description: 'Count'),
      );

      expect(stage.getParameter<int>('count'), equals(10));
      expect(() => stage.getParameter<double>('count'), throwsArgumentError);
    });

    test('provides fallback parameter access', () {
      stage.addParameter(
        Parameter.stageFloat('ratio', defaultValue: 1.0, description: 'Ratio'),
      );

      expect(stage.getParameterOr<double>('ratio', 2.0), equals(1.0));
      expect(stage.getParameterOr<double>('nonexistent', 2.0), equals(2.0));
      expect(
        stage.getParameterOr<int>('ratio', 5),
        equals(5),
      ); // Wrong type fallback
    });

    test('resets parameters to defaults', () async {
      stage.addParameter(
        Parameter.stageFloat('gain', defaultValue: 1.0, description: 'Gain'),
      );

      await stage.updateParameter('gain', 2.0);
      expect(stage.getParameter<double>('gain'), equals(2.0));

      await stage.resetParameter('gain');
      expect(stage.getParameter<double>('gain'), equals(1.0));
    });

    test('resets all parameters to defaults', () async {
      stage.addParameter(
        Parameter.stageFloat('gain', defaultValue: 1.0, description: 'Gain'),
      );
      stage.addParameter(
        Parameter.stageInt('delay', defaultValue: 100, description: 'Delay'),
      );

      await stage.updateParameters({'gain': 2.0, 'delay': 200});
      await stage.resetAllParameters();

      expect(stage.getParameter<double>('gain'), equals(1.0));
      expect(stage.getParameter<int>('delay'), equals(100));
    });

    test('filters parameters by type', () {
      stage.addParameter(
        Parameter.stageFloat('gain', defaultValue: 1.0, description: 'Gain'),
      );
      stage.addParameter(
        Parameter.stageInt('delay', defaultValue: 100, description: 'Delay'),
      );
      stage.addParameter(
        Parameter.stageFloat(
          'volume',
          defaultValue: 0.5,
          description: 'Volume',
        ),
      );

      final floatParams = stage.getParametersByType<double>();
      expect(floatParams, hasLength(2));
      expect(floatParams.keys, containsAll(['gain', 'volume']));

      final intParams = stage.getParametersByType<int>();
      expect(intParams, hasLength(1));
      expect(intParams.keys, contains('delay'));
    });

    test('validates all parameters', () async {
      stage.addParameter(
        Parameter.stageFloat(
          'gain',
          defaultValue: 1.0,
          min: 0.0,
          max: 2.0,
          description: 'Gain',
        ),
      );

      // Valid state with default values
      expect(stage.validateAllParameters().isValid, isTrue);

      // Set valid value - should still be valid
      await stage.updateParameter('gain', 1.5);
      expect(stage.validateAllParameters().isValid, isTrue);

      // Test validation during update - invalid value should throw
      expect(
        () async => await stage.updateParameter('gain', -1.0), // Below minimum
        throwsArgumentError,
      );

      // Parameter should still have valid value after failed update
      expect(stage.getParameter<double>('gain'), equals(1.5));
      expect(stage.validateAllParameters().isValid, isTrue);
    });

    test('tracks parameter modifications', () async {
      stage.addParameter(
        Parameter.stageFloat('gain', defaultValue: 1.0, description: 'Gain'),
      );
      stage.addParameter(
        Parameter.stageInt('delay', defaultValue: 100, description: 'Delay'),
      );

      expect(stage.hasModifiedParameters, isFalse);
      expect(stage.modifiedParameters, isEmpty);

      await stage.updateParameter('gain', 1.5);

      expect(stage.hasModifiedParameters, isTrue);
      expect(stage.modifiedParameters, contains('gain'));
      expect(stage.modifiedParameters, hasLength(1));
    });

    test('serializes and deserializes parameters', () async {
      stage.addParameter(
        Parameter.stageFloat('gain', defaultValue: 1.0, description: 'Gain'),
      );
      stage.addParameter(
        Parameter.stageInt('delay', defaultValue: 100, description: 'Delay'),
      );
      stage.addParameter(
        Parameter.stageBool(
          'enabled',
          defaultValue: true,
          description: 'Enabled',
        ),
      );

      await stage.updateParameters({
        'gain': 1.5,
        'delay': 200,
        'enabled': false,
      });

      final serialized = stage.serializeParameters();
      expect(serialized['gain'], equals('1.5'));
      expect(serialized['delay'], equals('200'));
      expect(serialized['enabled'], equals('false'));

      await stage.resetAllParameters();
      await stage.deserializeParameters(serialized);

      expect(stage.getParameter<double>('gain'), equals(1.5));
      expect(stage.getParameter<int>('delay'), equals(200));
      expect(stage.getParameter<bool>('enabled'), isFalse);
    });

    test('creates and restores parameter snapshots', () async {
      stage.addParameter(
        Parameter.stageFloat('gain', defaultValue: 1.0, description: 'Gain'),
      );
      stage.addParameter(
        Parameter.stageInt('delay', defaultValue: 100, description: 'Delay'),
      );

      await stage.updateParameters({'gain': 1.5, 'delay': 200});
      final snapshot = stage.createParameterSnapshot();

      await stage.resetAllParameters();
      expect(stage.getParameter<double>('gain'), equals(1.0));

      await stage.restoreParameterSnapshot(snapshot);
      expect(stage.getParameter<double>('gain'), equals(1.5));
      expect(stage.getParameter<int>('delay'), equals(200));
    });

    test('provides parameter introspection', () {
      stage.addParameter(
        Parameter.stageFloat(
          'gain',
          defaultValue: 1.0,
          min: 0.0,
          max: 2.0,
          description: 'Audio gain parameter',
        ),
      );

      final info = stage.getParameterInfo();
      expect(info, hasLength(1));
      expect(info['gain']?['description'], equals('Audio gain parameter'));
      expect(info['gain']?['defaultValue'], equals(1.0));
      expect(info['gain']?['min'], equals(0.0));
      expect(info['gain']?['max'], equals(2.0));
    });

    test('handles bulk parameter operations', () {
      final params = [
        Parameter.stageFloat('gain', defaultValue: 1.0, description: 'Gain'),
        Parameter.stageInt('delay', defaultValue: 100, description: 'Delay'),
        Parameter.stageBool(
          'enabled',
          defaultValue: true,
          description: 'Enabled',
        ),
      ];

      stage.addParameters(params);
      expect(stage.parameterDefinitions, hasLength(3));

      stage.clearParameters();
      expect(stage.parameterDefinitions, isEmpty);
    });

    test('calls parameter change callbacks', () async {
      var callbackCalled = false;
      var bulkCallbackCalled = false;

      stage.onParameterChangedCallback = (name, oldVal, newVal) async {
        expect(name, equals('gain'));
        expect(oldVal, equals(1.0));
        expect(newVal, equals(1.5));
        callbackCalled = true;
      };

      stage.onParametersChangedCallback = (oldValues, newValues) async {
        expect(newValues['delay'], equals(200));
        bulkCallbackCalled = true;
      };

      stage.addParameter(
        Parameter.stageFloat('gain', defaultValue: 1.0, description: 'Gain'),
      );
      stage.addParameter(
        Parameter.stageInt('delay', defaultValue: 100, description: 'Delay'),
      );

      await stage.updateParameter('gain', 1.5);
      expect(callbackCalled, isTrue);

      await stage.updateParameters({'delay': 200});
      expect(bulkCallbackCalled, isTrue);
    });
  });

  group('ParameterBuilder', () {
    late TestStage stage;

    setUp(() {
      stage = TestStage();
    });

    test('builds parameters with fluent API', () {
      stage
          .parameter('gain')
          .asFloat(1.0, min: 0.0, max: 2.0, description: 'Audio gain');

      stage
          .parameter('delay')
          .asInt(100, min: 0, max: 1000, description: 'Delay in milliseconds');

      stage.parameter('enabled').asBool(true, description: 'Enable processing');

      expect(stage.hasParameter('gain'), isTrue);
      expect(stage.hasParameter('delay'), isTrue);
      expect(stage.hasParameter('enabled'), isTrue);

      expect(stage.getParameter<double>('gain'), equals(1.0));
      expect(stage.getParameter<int>('delay'), equals(100));
      expect(stage.getParameter<bool>('enabled'), isTrue);
    });

    test('builds enum parameters', () {
      stage
          .parameter('color_scheme')
          .asEnum(
            TestColorScheme.red,
            TestColorScheme.values,
            description: 'Color scheme',
          );

      expect(stage.hasParameter('color_scheme'), isTrue);
      expect(
        stage.getParameter<TestColorScheme>('color_scheme'),
        equals(TestColorScheme.red),
      );
    });

    test('builds vector parameters', () {
      stage.parameter('position').asVector([
        1.0,
        2.0,
        3.0,
      ], description: '3D position');

      expect(stage.hasParameter('position'), isTrue);
      expect(
        stage.getParameter<List<double>>('position'),
        equals([1.0, 2.0, 3.0]),
      );
    });
  });
}

/// Test stage class for mixin testing
class TestStage with StageProperties {
  // Test callbacks for verifying parameter change notifications
  Future<void> Function(String, dynamic, dynamic)? onParameterChangedCallback;
  Future<void> Function(Map<String, dynamic>, Map<String, dynamic>)?
  onParametersChangedCallback;

  @override
  Future<void> onParameterChanged(
    String name,
    dynamic oldValue,
    dynamic newValue,
  ) async {
    await onParameterChangedCallback?.call(name, oldValue, newValue);
  }

  @override
  Future<void> onParametersChanged(
    Map<String, dynamic> oldValues,
    Map<String, dynamic> newValues,
  ) async {
    await onParametersChangedCallback?.call(oldValues, newValues);
  }
}

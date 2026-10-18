import 'package:gpu_tensor/gpu_tensor.dart';

/// Result of input validation
class ValidationResult {
  final bool isValid;
  final List<String> errors;
  final List<String> warnings;

  const ValidationResult._({
    required this.isValid,
    this.errors = const [],
    this.warnings = const [],
  });

  factory ValidationResult.valid({List<String> warnings = const []}) {
    return ValidationResult._(isValid: true, warnings: warnings);
  }

  factory ValidationResult.invalid(String error) {
    return ValidationResult._(isValid: false, errors: [error]);
  }

  factory ValidationResult.invalidMultiple(List<String> errors) {
    return ValidationResult._(isValid: false, errors: errors);
  }

  @override
  String toString() {
    if (isValid) {
      return warnings.isEmpty
          ? 'Valid'
          : 'Valid with warnings: ${warnings.join(', ')}';
    } else {
      return 'Invalid: ${errors.join(', ')}';
    }
  }
}

/// Shape validation policies
enum InputShapePolicy {
  flexible, // Accept any shape
  exact, // Must match exact dimensions
  constrained, // Min/max constraints
  custom, // Custom validation
}

enum OutputSizePolicy {
  fixed, // Constant size
  inputDependent, // Based on input size
  parameterDependent, // Based on parameters
  computed, // Custom calculation
}

/// Base class for all ports
abstract class Port {
  final String name;
  final String description;
  final String format;
  final bool persistent;

  const Port({
    required this.name,
    this.description = '',
    this.format = 'tensor',
    this.persistent = false,
  });

  @override
  String toString() => 'Port($name, format: $format)';

  /// Factory for deserialization
  static Port fromConfig(Map<String, dynamic> config) {
    final type = config['type'] as String?;
    switch (type) {
      case 'flexible_input':
        return FlexibleInputPort(
          name: config['name'],
          description: config['description'] ?? '',
          isRequired: config['isRequired'] ?? true,
          minDimensions: config['minDimensions'],
          maxDimensions: config['maxDimensions'],
        );
      case 'exact_input':
        return ExactInputPort(
          name: config['name'],
          expectedShape: List<int>.from(
            config['expectedShape'] ?? config['shape'] ?? [],
          ),
          description: config['description'] ?? '',
        );
      case 'audio_input':
        return AudioInputPort(
          name: config['name'],
          description: config['description'] ?? '',
          isRequired: config['isRequired'] ?? true,
          minSamples: config['minSamples'],
          maxSamples: config['maxSamples'],
          expectedChannels: config['expectedChannels'],
        );
      case 'video_input':
        return VideoInputPort(
          name: config['name'],
          description: config['description'] ?? '',
          isRequired: config['isRequired'] ?? true,
          minWidth: config['minWidth'],
          maxWidth: config['maxWidth'],
          minHeight: config['minHeight'],
          maxHeight: config['maxHeight'],
        );
      case 'static_input':
        return StaticInputPort(
          name: config['name'],
          assetPath: config['assetPath'],
          description: config['description'] ?? '',
        );
      case 'generated_input':
        return GeneratedInputPort(
          name: config['name'],
          shape: List<int>.from(config['shape'] ?? []),
          generatorType: config['generatorType'] ?? 'noise',
          generatorParameters: config['generatorParameters'] ?? {},
          description: config['description'] ?? '',
        );
      default:
        throw Exception('Unknown port type: $type');
    }
  }
}

// OutputPort factory
abstract class OutputPort extends Port {
  final OutputSizePolicy sizePolicy;

  const OutputPort({
    required super.name,
    super.description = '',
    super.format = 'tensor',
    super.persistent = false,
    this.sizePolicy = OutputSizePolicy.inputDependent,
  });

  List<int> calculateOutputShape(
    Map<String, Tensor> inputs,
    Map<String, dynamic> parameters,
  );

  /// Factory for deserialization
  static OutputPort fromConfig(Map<String, dynamic> config) {
    final type = config['type'] as String?;
    switch (type) {
      case 'fixed_output':
        return FixedOutputPort(
          name: config['name'],
          shape: List<int>.from(config['shape'] ?? []),
          description: config['description'] ?? '',
          format: config['format'] ?? 'tensor',
        );
      case 'dynamic_output':
        // If fixedShape is present, treat as fixed
        if (config['fixedShape'] != null) {
          return DynamicOutputPort(
            name: config['name'],
            fixedShape: List<int>.from(config['fixedShape']),
            description: config['description'] ?? '',
            format: config['format'] ?? 'tensor',
          );
        }
        // Otherwise, provide a dummy shapeCalculator (can be replaced after construction)
        return DynamicOutputPort(
          name: config['name'],
          shapeCalculator: (inputs, parameters) => inputs.isNotEmpty
              ? List<int>.from(inputs.values.first.shape)
              : [1],
          description: config['description'] ?? '',
          format: config['format'] ?? 'tensor',
        );
      case 'texture_output':
        return TextureOutputPort(
          name: config['name'],
          width: config['width'],
          height: config['height'],
          channels: config['channels'] ?? 4,
          description: config['description'] ?? '',
        );
      case 'history_output':
        return HistoryOutputPort(
          name: config['name'],
          timeSlices: config['timeSlices'],
          description: config['description'] ?? '',
        );
      case 'parameter_output':
        return ParameterOutputPort(
          name: config['name'],
          parameterDependencies: List<String>.from(
            config['parameterDependencies'] ?? [],
          ),
          shapeCalculator: (inputs, parameters) => inputs.isNotEmpty
              ? List<int>.from(inputs.values.first.shape)
              : [1],
          description: config['description'] ?? '',
          format: config['format'] ?? 'tensor',
        );
      default:
        throw Exception('Unknown output port type: $type');
    }
  }
}

/// Input port base class
abstract class InputPort extends Port {
  final bool isRequired;
  final InputShapePolicy shapePolicy;

  const InputPort({
    required super.name,
    super.description = '',
    super.format = 'tensor',
    super.persistent = false,
    this.isRequired = true,
    this.shapePolicy = InputShapePolicy.flexible,
  });

  ValidationResult validateInput(Tensor? tensor);
  List<int>? get expectedShape => null;
}

/// Flexible input that accepts any tensor shape
class FlexibleInputPort extends InputPort {
  final int? minDimensions;
  final int? maxDimensions;

  const FlexibleInputPort({
    required super.name,
    super.description = '',
    super.isRequired = true,
    this.minDimensions,
    this.maxDimensions,
  });

  @override
  ValidationResult validateInput(Tensor? tensor) {
    if (tensor == null) {
      return isRequired
          ? ValidationResult.invalid('Input $name is required')
          : ValidationResult.valid();
    }

    if (minDimensions != null && tensor.shape.length < minDimensions!) {
      return ValidationResult.invalid(
        'Input $name needs at least $minDimensions dimensions',
      );
    }

    if (maxDimensions != null && tensor.shape.length > maxDimensions!) {
      return ValidationResult.invalid(
        'Input $name has too many dimensions (max $maxDimensions)',
      );
    }

    return ValidationResult.valid();
  }
}

/// Audio input with sample constraints
class AudioInputPort extends InputPort {
  final int? minSamples;
  final int? maxSamples;
  final int? expectedChannels;

  const AudioInputPort({
    required super.name,
    super.description = '',
    super.isRequired = true,
    this.minSamples,
    this.maxSamples,
    this.expectedChannels,
  });

  @override
  ValidationResult validateInput(Tensor? tensor) {
    if (tensor == null) {
      return isRequired
          ? ValidationResult.invalid('Audio input $name is required')
          : ValidationResult.valid();
    }

    if (tensor.shape.length > 2) {
      return ValidationResult.invalid('Audio must be 1D or 2D tensor');
    }

    final samples = tensor.shape.length == 1
        ? tensor.shape[0]
        : tensor.shape[1];
    final channels = tensor.shape.length == 1 ? 1 : tensor.shape[0];

    if (minSamples != null && samples < minSamples!) {
      return ValidationResult.invalid('Need at least $minSamples samples');
    }

    if (maxSamples != null && samples > maxSamples!) {
      return ValidationResult.invalid('Too many samples (max $maxSamples)');
    }

    if (expectedChannels != null && channels != expectedChannels!) {
      return ValidationResult.invalid(
        'Expected $expectedChannels channels, got $channels',
      );
    }

    return ValidationResult.valid();
  }
}

/// Video input with resolution constraints
class VideoInputPort extends InputPort {
  final int? minWidth;
  final int? maxWidth;
  final int? minHeight;
  final int? maxHeight;

  const VideoInputPort({
    required super.name,
    super.description = '',
    super.isRequired = true,
    this.minWidth,
    this.maxWidth,
    this.minHeight,
    this.maxHeight,
  });

  @override
  ValidationResult validateInput(Tensor? tensor) {
    if (tensor == null) {
      return isRequired
          ? ValidationResult.invalid('Video input $name is required')
          : ValidationResult.valid();
    }

    if (tensor.shape.length != 3 && tensor.shape.length != 4) {
      return ValidationResult.invalid(
        'Video must be 3D [H,W,C] or 4D [B,H,W,C]',
      );
    }

    final isBatched = tensor.shape.length == 4;
    final height = tensor.shape[isBatched ? 1 : 0];
    final width = tensor.shape[isBatched ? 2 : 1];

    if (minWidth != null && width < minWidth!) {
      return ValidationResult.invalid('Width too small (min $minWidth)');
    }
    if (maxWidth != null && width > maxWidth!) {
      return ValidationResult.invalid('Width too large (max $maxWidth)');
    }
    if (minHeight != null && height < minHeight!) {
      return ValidationResult.invalid('Height too small (min $minHeight)');
    }
    if (maxHeight != null && height > maxHeight!) {
      return ValidationResult.invalid('Height too large (max $maxHeight)');
    }

    return ValidationResult.valid();
  }
}

/// Input with exact shape requirements
class ExactInputPort extends InputPort {
  @override
  final List<int> expectedShape;

  const ExactInputPort({
    required super.name,
    required this.expectedShape,
    super.description = '',
  }) : super(shapePolicy: InputShapePolicy.exact);

  @override
  ValidationResult validateInput(Tensor? tensor) {
    if (tensor == null) {
      return ValidationResult.invalid('Input $name is required');
    }

    if (tensor.shape.length != expectedShape.length ||
        !_listsEqual(tensor.shape, expectedShape)) {
      return ValidationResult.invalid(
        'Shape ${tensor.shape} != expected $expectedShape',
      );
    }

    return ValidationResult.valid();
  }

  bool _listsEqual(List<int> a, List<int> b) {
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }
}

/// Input with custom validation
class CustomInputPort extends InputPort {
  final ValidationResult Function(Tensor?) validator;

  const CustomInputPort({
    required super.name,
    required this.validator,
    super.isRequired = true,
    super.description = '',
  }) : super(shapePolicy: InputShapePolicy.custom);

  @override
  ValidationResult validateInput(Tensor? tensor) {
    if (tensor == null && !isRequired) return ValidationResult.valid();
    return validator(tensor);
  }
}

/// Static asset input
class StaticInputPort extends InputPort {
  final String assetPath;

  const StaticInputPort({
    required super.name,
    required this.assetPath,
    super.description = '',
    super.persistent = true,
  });

  @override
  ValidationResult validateInput(Tensor? tensor) {
    return tensor == null
        ? ValidationResult.invalid('Asset $name not loaded')
        : ValidationResult.valid();
  }
}

/// Fixed size output
class FixedOutputPort extends OutputPort {
  final List<int> shape;

  const FixedOutputPort({
    required super.name,
    required this.shape,
    super.description = '',
    super.format = 'tensor',
  }) : super(sizePolicy: OutputSizePolicy.fixed);

  @override
  List<int> calculateOutputShape(
    Map<String, Tensor> inputs,
    Map<String, dynamic> parameters,
  ) {
    return List.from(shape);
  }
}

/// Dynamic output with custom calculation
class DynamicOutputPort extends OutputPort {
  final List<int> Function(Map<String, Tensor>, Map<String, dynamic>)?
  shapeCalculator;
  final List<int>? fixedShape;

  const DynamicOutputPort({
    required super.name,
    super.description = '',
    super.format = 'tensor',
    super.persistent = false,
    this.shapeCalculator,
    this.fixedShape,
  }) : super(
         sizePolicy: shapeCalculator != null
             ? OutputSizePolicy.computed
             : OutputSizePolicy.fixed,
       );

  @override
  List<int> calculateOutputShape(
    Map<String, Tensor> inputs,
    Map<String, dynamic> parameters,
  ) {
    if (fixedShape != null) return List.from(fixedShape!);
    if (shapeCalculator != null) return shapeCalculator!(inputs, parameters);
    if (inputs.isNotEmpty) return List.from(inputs.values.first.shape);
    throw Exception('Cannot calculate output shape for $name');
  }
}

/// Texture output (RGBA image)
class TextureOutputPort extends FixedOutputPort {
  final int width;
  final int height;
  final int channels;

  TextureOutputPort({
    required super.name,
    super.description = '',
    required this.width,
    required this.height,
    this.channels = 4,
  }) : super(shape: [width * height * channels], format: 'rgba_texture');

  List<int> get textureDimensions => [width, height, channels];
  int get pixelCount => width * height;
}

/// History buffer output for temporal data
class HistoryOutputPort extends DynamicOutputPort {
  final int timeSlices;

  HistoryOutputPort({
    required super.name,
    required this.timeSlices,
    required super.description,
  }) : super(format: 'history_buffer');

  @override
  List<int> calculateOutputShape(
    Map<String, Tensor> inputs,
    Map<String, dynamic> parameters,
  ) {
    if (inputs.isNotEmpty) {
      return [timeSlices, ...inputs.values.first.shape];
    }
    return [timeSlices];
  }
}

/// Input port collection
class InputPortCollection {
  final Map<String, InputPort> _ports = {};

  void add(InputPort port) => _ports[port.name] = port;

  /// Clear all ports
  void clear() => _ports.clear();

  InputPort? operator [](String name) => _ports[name];
  List<InputPort> get all => _ports.values.toList();
  List<String> get names => _ports.keys.toList();
  List<InputPort> get required =>
      _ports.values.where((p) => p.isRequired).toList();

  ValidationResult validateAll(Map<String, Tensor> inputs) {
    final errors = <String>[];
    final warnings = <String>[];

    for (final port in _ports.values) {
      final input = inputs[port.name];
      final result = port.validateInput(input);

      if (!result.isValid) {
        errors.addAll(result.errors);
      }
      warnings.addAll(result.warnings);
    }

    return errors.isEmpty
        ? ValidationResult.valid(warnings: warnings)
        : ValidationResult.invalidMultiple(errors);
  }

  int get length => _ports.length;
  bool get isEmpty => _ports.isEmpty;
}

/// Output port collection
class OutputPortCollection {
  final Map<String, OutputPort> _ports = {};

  void add(OutputPort port) => _ports[port.name] = port;

  /// Clear all ports
  void clear() => _ports.clear();

  OutputPort? operator [](String name) => _ports[name];
  List<OutputPort> get all => _ports.values.toList();
  List<String> get names => _ports.keys.toList();

  Future<Map<String, Tensor>> createOutputTensors(
    Map<String, Tensor> inputs,
    Map<String, dynamic> parameters,
  ) async {
    final outputs = <String, Tensor>{};
    for (final port in _ports.values) {
      final shape = port.calculateOutputShape(inputs, parameters);
      final tensor = await Tensor.create(shape);
      outputs[port.name] = tensor;
    }
    return outputs;
  }

  int get length => _ports.length;
  bool get isEmpty => _ports.isEmpty;
}

/// Generated input for synthetic data
class GeneratedInputPort extends InputPort {
  final List<int> shape;
  final String generatorType; // 'noise', 'pattern', 'zeros', 'ones'
  final Map<String, dynamic> generatorParameters;

  const GeneratedInputPort({
    required super.name,
    required this.shape,
    required this.generatorType,
    this.generatorParameters = const {},
    super.description,
  }) : super(format: 'generated');

  @override
  ValidationResult validateInput(Tensor? tensor) {
    // Generated inputs are always valid when created
    return ValidationResult.valid();
  }

  @override
  List<int> get expectedShape => shape;
}

/// Parameter-dependent output port
class ParameterOutputPort extends DynamicOutputPort {
  final List<String> parameterDependencies;

  const ParameterOutputPort({
    required super.name,
    required this.parameterDependencies,
    required List<int> Function(Map<String, Tensor>, Map<String, dynamic>)
    super.shapeCalculator,
    super.description,
    super.format,
  });

  /// Validates that all required parameters are present
  ValidationResult validateParameters(Map<String, dynamic> parameters) {
    final missing = <String>[];
    for (final param in parameterDependencies) {
      if (!parameters.containsKey(param)) {
        missing.add(param);
      }
    }

    return missing.isNotEmpty
        ? ValidationResult.invalid(
            'Output $name missing required parameters: ${missing.join(', ')}',
          )
        : ValidationResult.valid();
  }
}

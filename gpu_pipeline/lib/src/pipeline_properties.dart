import 'package:gpu_pipeline/src/pipeline_ports.dart';

/// Registry for enum types to support serialization/deserialization
class EnumRegistry {
  static final Map<String, List<Enum>> _enumTypes = {};
  static final Map<String, Map<String, Enum>> _enumValuesByName = {};

  /// Register an enum type with all its values
  static void register<T extends Enum>(List<T> values) {
    if (values.isEmpty) return;

    final typeName = T.toString();
    _enumTypes[typeName] = values.cast<Enum>();

    // Create name-to-value mapping for this enum
    final nameMap = <String, Enum>{};
    for (final value in values) {
      nameMap[value.toString()] = value;
    }
    _enumValuesByName[typeName] = nameMap;
  }

  /// Get all values for an enum type
  static List<Enum>? getValues(String enumTypeName) {
    return _enumTypes[enumTypeName];
  }

  /// Get enum value by type name and string representation
  static Enum? getValueByName(String enumTypeName, String valueName) {
    return _enumValuesByName[enumTypeName]?[valueName];
  }

  /// Get enum value by type name and index
  static Enum? getValueByIndex(String enumTypeName, int index) {
    final values = _enumTypes[enumTypeName];
    if (values == null || index < 0 || index >= values.length) return null;
    return values[index];
  }

  /// Check if an enum type is registered
  static bool isRegistered(String enumTypeName) {
    return _enumTypes.containsKey(enumTypeName);
  }

  static void clear() {
    _enumTypes.clear();
    _enumValuesByName.clear();
  }

  /// Get all registered enum type names
  static Set<String> get registeredTypes => _enumTypes.keys.toSet();
}

class Parameter<T> {
  final String name;
  final T defaultValue;
  final T? min;
  final T? max;
  final List<T>? allowedValues;
  final String description;

  Parameter({
    required this.name,
    required this.defaultValue,
    this.min,
    this.max,
    this.allowedValues,
    required this.description,
  });

  /// Validate a value against this parameter's constraints
  ValidationResult validate(dynamic value) {
    if (value is! T) {
      return ValidationResult.invalid(
        'Expected ${T.toString()}, got ${value.runtimeType}',
      );
    }

    // Range validation - fix Comparable comparison
    if (min != null && value is Comparable) {
      if (value.compareTo(min) < 0) {
        return ValidationResult.invalid('Value $value below minimum $min');
      }
    }
    if (max != null && value is Comparable) {
      if (value.compareTo(max) > 0) {
        return ValidationResult.invalid('Value $value above maximum $max');
      }
    }

    // Allowed values validation
    if (allowedValues != null && !allowedValues!.contains(value)) {
      return ValidationResult.invalid(
        'Value $value not in allowed values: $allowedValues',
      );
    }

    return ValidationResult.valid();
  }

  // Also fix the sanitize() method:
  T sanitize(dynamic value) {
    if (value is! T) return defaultValue;

    // Clamp to range if numeric - fix comparison
    if (value is Comparable) {
      if (min != null && value.compareTo(min) < 0) return min as T;
      if (max != null && value.compareTo(max) > 0) return max as T;
    }

    // Use allowed values if restricted
    if (allowedValues != null && !allowedValues!.contains(value)) {
      return defaultValue;
    }

    return value;
  }

  /// Serialize parameter value to string
  String serialize(T value) {
    if (T == bool || T == int || T == double || T == String) {
      return value.toString();
    }
    if (value is Enum) {
      return '${value.runtimeType}:${value.name}';
    }
    if (value is List) {
      return value.map((e) => e.toString()).join(',');
    }
    return value.toString();
  }

  /// Deserialize parameter value from string
  T deserialize(String data) {
    try {
      if (T == bool) return (data.toLowerCase() == 'true') as T;
      if (T == int) return int.parse(data) as T;
      if (T == double) return double.parse(data) as T;
      if (T == String) return data as T;

      // Enum deserialization
      if (data.contains(':')) {
        final parts = data.split(':');
        final enumValue = EnumRegistry.getValueByName(parts[0], parts[1]);
        if (enumValue != null) return enumValue as T;
      }

      // List deserialization (basic)
      if (T.toString().startsWith('List<')) {
        final items = data.split(',');
        // This would need more sophisticated type handling
        return items as T;
      }
    } catch (e) {
      // Fall back to default on parse error
    }

    return defaultValue;
  }

  // Factory constructors for common types
  static Parameter<double> stageFloat(
    String name, {
    required double defaultValue,
    double? min,
    double? max,
    String? description,
  }) => Parameter<double>(
    name: name,
    defaultValue: defaultValue,
    min: min,
    max: max,
    description: description ?? '',
  );

  static Parameter<int> stageInt(
    String name, {
    required int defaultValue,
    int? min,
    int? max,
    String? description,
  }) => Parameter<int>(
    name: name,
    defaultValue: defaultValue,
    min: min,
    max: max,
    description: description ?? ' ',
  );

  static Parameter<bool> stageBool(
    String name, {
    required bool defaultValue,
    String? description,
  }) => Parameter<bool>(
    name: name,
    defaultValue: defaultValue,
    description: description ?? ' ',
  );

  static Parameter<String> stageString(
    String name, {
    required String defaultValue,
    List<String>? allowedValues,
    String? description,
  }) => Parameter<String>(
    name: name,
    defaultValue: defaultValue,
    allowedValues: allowedValues,
    description: description ?? ' ',
  );

  /// Enhanced enum parameter with automatic registration
  static Parameter<T> stageEnum<T extends Enum>(
    String name, {
    required T defaultValue,
    required List<T> allValues, // Now required - provides all enum values
    List<T>? allowedValues, // Optional subset of allowed values
    String? description,
  }) {
    // Auto-register the enum type
    EnumRegistry.register(allValues);

    return Parameter<T>(
      name: name,
      defaultValue: defaultValue,
      allowedValues:
          allowedValues ?? allValues, // Use all values if no restriction
      description: description ?? ' ',
    );
  }

  // Vector parameters for 3D graphics
  static Parameter<List<double>> stageVec2(
    String name, {
    required List<double> defaultValue,
    String? description,
  }) {
    assert(defaultValue.length == 2, 'Vec2 must have exactly 2 components');
    return Parameter<List<double>>(
      name: name,
      defaultValue: defaultValue,
      description: description ?? ' ',
    );
  }

  static Parameter<List<double>> stageVec3(
    String name, {
    required List<double> defaultValue,
    String? description,
  }) {
    assert(defaultValue.length == 3, 'Vec3 must have exactly 3 components');
    return Parameter<List<double>>(
      name: name,
      defaultValue: defaultValue,
      description: description ?? ' ',
    );
  }

  static Parameter<List<double>> stageVec4(
    String name, {
    required List<double> defaultValue,
    String? description,
  }) {
    assert(defaultValue.length == 4, 'Vec4 must have exactly 4 components');
    return Parameter<List<double>>(
      name: name,
      defaultValue: defaultValue,
      description: description ?? ' ',
    );
  }

  /// Create a float list parameter for pipeline stages
  static Parameter<List<double>> stageFloatList(
    String name, {
    required List<double> defaultValue,
    String? description,
    List<double>? min,
    List<double>? max,
  }) {
    return Parameter<List<double>>(
      name: name,
      defaultValue: defaultValue,
      description: description ?? 'Float list parameter',
    );
  }
}

/// Mixin to add parameter management to pipeline stages
mixin StageProperties {
  final Map<String, Parameter> _parameterDefinitions = {};
  final Map<String, dynamic> _currentValues = {};

  /// Add a parameter definition to this stage
  void addParameter<T>(Parameter<T> parameter) {
    _parameterDefinitions[parameter.name] = parameter;
    _currentValues[parameter.name] = parameter.defaultValue;
  }

  /// Get all parameter definitions
  Map<String, Parameter> get parameterDefinitions =>
      Map.unmodifiable(_parameterDefinitions);

  /// Get current parameter values
  Map<String, dynamic> get currentParameters =>
      Map.unmodifiable(_currentValues);

  /// Update a parameter value with validation
  Future<void> updateParameter(String name, dynamic value) async {
    final parameter = _parameterDefinitions[name];
    if (parameter == null) {
      throw ArgumentError('Parameter $name not found');
    }

    final validation = parameter.validate(value);
    if (!validation.isValid) {
      throw ArgumentError(
        'Invalid parameter value: ${validation.errors.join(', ')}',
      );
    }

    final oldValue = _currentValues[name];
    _currentValues[name] = parameter.sanitize(value);

    // Notify parameter change
    await onParameterChanged(name, oldValue, _currentValues[name]);
  }

  /// Update multiple parameters atomically
  Future<void> updateParameters(Map<String, dynamic> updates) async {
    // Validate all parameters first
    final validationErrors = <String>[];
    final sanitizedUpdates = <String, dynamic>{};

    for (final entry in updates.entries) {
      final parameter = _parameterDefinitions[entry.key];
      if (parameter == null) {
        validationErrors.add('Parameter ${entry.key} not found');
        continue;
      }

      final validation = parameter.validate(entry.value);
      if (!validation.isValid) {
        validationErrors.addAll(
          validation.errors.map((e) => '${entry.key}: $e'),
        );
      } else {
        sanitizedUpdates[entry.key] = parameter.sanitize(entry.value);
      }
    }

    if (validationErrors.isNotEmpty) {
      throw ArgumentError(
        'Parameter validation failed:\n${validationErrors.join('\n')}',
      );
    }

    // Apply all updates atomically
    final oldValues = <String, dynamic>{};
    for (final entry in sanitizedUpdates.entries) {
      oldValues[entry.key] = _currentValues[entry.key];
      _currentValues[entry.key] = entry.value;
    }

    await onParametersChanged(oldValues, sanitizedUpdates);
  }

  /// Check if a parameter exists
  bool hasParameter(String name) => _parameterDefinitions.containsKey(name);

  /// Check if a parameter value is valid
  bool isValidParameterValue(String name, dynamic value) {
    final parameter = _parameterDefinitions[name];
    if (parameter == null) return false;
    return parameter.validate(value).isValid;
  }

  /// Get parameter value with type safety
  T getParameter<T>(String name) {
    if (!_parameterDefinitions.containsKey(name)) {
      throw ArgumentError('Parameter $name not found');
    }

    final value = _currentValues[name];
    if (value is T) return value;

    final parameter = _parameterDefinitions[name]!;
    if (parameter.defaultValue is T) {
      return parameter.defaultValue as T;
    }

    throw ArgumentError(
      'Parameter $name is not of type $T, got ${value.runtimeType}',
    );
  }

  /// Set parameter value with validation
  Future<void> setParameter(String name, dynamic value) async {
    final parameter = _parameterDefinitions[name];
    if (parameter == null) {
      throw Exception('Parameter "$name" not found');
    }

    final validationResult = parameter.validate(value);
    if (!validationResult.isValid) {
      throw Exception(
        'Invalid value for parameter "$name": ${validationResult.errors.join(', ')}',
      );
    }

    final oldValue = _currentValues[name];
    _currentValues[name] = value;

    await onParameterChanged(name, oldValue, value);
  }

  /// Get parameter value with fallback
  T getParameterOr<T>(String name, T fallback) {
    try {
      return getParameter<T>(name);
    } catch (_) {
      return fallback;
    }
  }

  /// Set property value (alias for backward compatibility)
  Future<bool> setPropertyValue(String name, dynamic value) async {
    try {
      await setParameter(name, value);
      return true;
    } catch (e) {
      return false;
    }
  }

  /// Get property value (alias for backward compatibility)
  T? getPropertyValue<T>(String name) {
    return getParameter<T>(name);
  }

  /// Get property metadata (for backward compatibility)
  Map<String, Parameter> get propertyMetadata => parameterDefinitions;

  /// Bulk parameter update
  Future<void> setParameters(Map<String, dynamic> parameters) async {
    final oldValues = Map<String, dynamic>.from(_currentValues);

    for (final entry in parameters.entries) {
      await setParameter(entry.key, entry.value);
    }

    await onParametersChanged(oldValues, Map.from(_currentValues));
  }

  /// Get parameter definition by name
  Parameter? getParameterDefinition(String name) => _parameterDefinitions[name];

  /// Reset parameter to default value
  Future<void> resetParameter(String name) async {
    final parameter = _parameterDefinitions[name];
    if (parameter == null) {
      throw ArgumentError('Parameter $name not found');
    }

    await updateParameter(name, parameter.defaultValue);
  }

  /// Reset all parameters to default values
  Future<void> resetAllParameters() async {
    final defaults = <String, dynamic>{};
    for (final param in _parameterDefinitions.values) {
      defaults[param.name] = param.defaultValue;
    }
    await updateParameters(defaults);
  }

  /// Get parameters by type
  Map<String, T> getParametersByType<T>() {
    final result = <String, T>{};
    for (final entry in _currentValues.entries) {
      if (entry.value is T) {
        result[entry.key] = entry.value as T;
      }
    }
    return result;
  }

  /// Validate all current parameter values
  ValidationResult validateAllParameters() {
    final errors = <String>[];
    final warnings = <String>[];

    for (final entry in _currentValues.entries) {
      final parameter = _parameterDefinitions[entry.key];
      if (parameter != null) {
        final result = parameter.validate(entry.value);
        if (!result.isValid) {
          errors.addAll(result.errors.map((e) => '${entry.key}: $e'));
        }
        warnings.addAll(result.warnings.map((w) => '${entry.key}: $w'));
      }
    }

    if (errors.isNotEmpty) {
      return ValidationResult.invalidMultiple(errors);
    }

    return ValidationResult.valid(warnings: warnings);
  }

  /// Override to handle single parameter changes
  Future<void> onParameterChanged(
    String name,
    dynamic oldValue,
    dynamic newValue,
  ) async {
    // Default implementation - stages can override
  }

  /// Override to handle bulk parameter changes
  Future<void> onParametersChanged(
    Map<String, dynamic> oldValues,
    Map<String, dynamic> newValues,
  ) async {
    // Default implementation - stages can override
    // Can be used for optimized bulk updates
  }

  /// Serialize all parameters to a map
  Map<String, String> serializeParameters() {
    final result = <String, String>{};
    for (final entry in _currentValues.entries) {
      final parameter = _parameterDefinitions[entry.key];
      if (parameter != null) {
        result[entry.key] = parameter.serialize(entry.value);
      }
    }
    return result;
  }

  /// Serialize specific parameters to a map
  Map<String, String> serializeParametersSubset(List<String> parameterNames) {
    final result = <String, String>{};
    for (final name in parameterNames) {
      final parameter = _parameterDefinitions[name];
      if (parameter != null && _currentValues.containsKey(name)) {
        result[name] = parameter.serialize(_currentValues[name]);
      }
    }
    return result;
  }

  /// Deserialize parameters from a map
  Future<void> deserializeParameters(Map<String, String> data) async {
    final updates = <String, dynamic>{};
    final errors = <String>[];

    for (final entry in data.entries) {
      final parameter = _parameterDefinitions[entry.key];
      if (parameter != null) {
        try {
          updates[entry.key] = parameter.deserialize(entry.value);
        } catch (e) {
          errors.add('Failed to deserialize ${entry.key}: $e');
        }
      } else {
        errors.add('Unknown parameter: ${entry.key}');
      }
    }

    if (errors.isNotEmpty) {
      throw ArgumentError('Deserialization failed:\n${errors.join('\n')}');
    }

    if (updates.isNotEmpty) {
      await updateParameters(updates);
    }
  }

  /// Create a snapshot of current parameter state
  Map<String, dynamic> createParameterSnapshot() {
    return Map<String, dynamic>.from(_currentValues);
  }

  /// Restore parameters from a snapshot
  Future<void> restoreParameterSnapshot(Map<String, dynamic> snapshot) async {
    // Only restore parameters that exist in current definitions
    final validUpdates = <String, dynamic>{};
    for (final entry in snapshot.entries) {
      if (_parameterDefinitions.containsKey(entry.key)) {
        validUpdates[entry.key] = entry.value;
      }
    }

    if (validUpdates.isNotEmpty) {
      await updateParameters(validUpdates);
    }
  }

  /// Get parameter info for debugging/introspection
  Map<String, Map<String, dynamic>> getParameterInfo() {
    final info = <String, Map<String, dynamic>>{};

    for (final param in _parameterDefinitions.values) {
      info[param.name] = {
        'type': param.runtimeType.toString(),
        'description': param.description,
        'defaultValue': param.defaultValue,
        'currentValue': _currentValues[param.name],
        'min': param.min,
        'max': param.max,
        'allowedValues': param.allowedValues,
      };
    }

    return info;
  }

  /// Check if any parameters have been modified from defaults
  bool get hasModifiedParameters {
    for (final param in _parameterDefinitions.values) {
      if (_currentValues[param.name] != param.defaultValue) {
        return true;
      }
    }
    return false;
  }

  /// Get list of modified parameters
  List<String> get modifiedParameters {
    final modified = <String>[];
    for (final param in _parameterDefinitions.values) {
      if (_currentValues[param.name] != param.defaultValue) {
        modified.add(param.name);
      }
    }
    return modified;
  }

  /// Clear all parameter definitions and values
  void clearParameters() {
    _parameterDefinitions.clear();
    _currentValues.clear();
  }

  /// Add multiple parameters at once
  void addParameters(List<Parameter> parameters) {
    for (final param in parameters) {
      addParameter(param);
    }
  }

  /// Create parameter builder for fluent API
  ParameterBuilder parameter(String name) => ParameterBuilder(name, this);
}

/// Fluent builder for adding parameters to stages
class ParameterBuilder {
  final String name;
  final StageProperties stage;

  ParameterBuilder(this.name, this.stage);

  /// Add float parameter
  ParameterBuilder asFloat(
    double defaultValue, {
    double? min,
    double? max,
    String? description,
  }) {
    stage.addParameter(
      Parameter.stageFloat(
        name,
        defaultValue: defaultValue,
        min: min,
        max: max,
        description: description ?? ' ',
      ),
    );
    return this;
  }

  /// Add int parameter
  ParameterBuilder asInt(
    int defaultValue, {
    int? min,
    int? max,
    String? description,
  }) {
    stage.addParameter(
      Parameter.stageInt(
        name,
        defaultValue: defaultValue,
        min: min,
        max: max,
        description: description ?? ' ',
      ),
    );
    return this;
  }

  /// Add bool parameter
  ParameterBuilder asBool(bool defaultValue, {required description}) {
    stage.addParameter(
      Parameter.stageBool(
        name,
        defaultValue: defaultValue,
        description: description ?? ' ',
      ),
    );
    return this;
  }

  /// Add enum parameter
  ParameterBuilder asEnum<T extends Enum>(
    T defaultValue,
    List<T> allValues, {
    String? description,
  }) {
    stage.addParameter(
      Parameter.stageEnum(
        name,
        defaultValue: defaultValue,
        allValues: allValues,
        description: description ?? ' ',
      ),
    );
    return this;
  }

  /// Add vector parameter
  ParameterBuilder asVector(List<double> defaultValue, {String? description}) {
    stage.addParameter(
      Parameter(
        name: name,
        defaultValue: defaultValue,
        description: description ?? ' ',
      ),
    );
    return this;
  }
}

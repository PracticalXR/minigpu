import 'dart:async';
import 'dart:math';
import 'dart:typed_data';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:minigpu/minigpu.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Minigpu Type Testing Lab',
      theme: ThemeData.dark().copyWith(
        primaryColor: Color(0xFF00FFFF), // Electric cyan
        colorScheme: ColorScheme.dark(
          primary: Color(0xFF00FFFF),
          secondary: Color(0xFF0080FF),
          surface: Color(0xFF1A1A2E),
          background: Color(0xFF0F0F23),
        ),
        cardColor: Color(0xFF1A1A2E),
        appBarTheme: AppBarTheme(
          backgroundColor: Color(0xFF0F0F23),
          foregroundColor: Color(0xFF00FFFF),
        ),
      ),
      home: TypeTestingExample(),
    );
  }
}

enum DataTypeDemo {
  int8('Int8', BufferDataType.int8, -128, 127),
  uint8('Uint8', BufferDataType.uint8, 0, 255),
  int16('Int16', BufferDataType.int16, -32768, 32767),
  uint16('Uint16', BufferDataType.uint16, 0, 65535),
  int32('Int32', BufferDataType.int32, -2147483648, 2147483647),
  uint32('Uint32', BufferDataType.uint32, 0, 4294967295),
  float32('Float32', BufferDataType.float32, -1000.0, 1000.0),
  float64('Float64', BufferDataType.float64, -1000.0, 1000.0);

  const DataTypeDemo(this.label, this.type, this.minValue, this.maxValue);

  final String label;
  final BufferDataType type;
  final double minValue;
  final double maxValue;
}

enum AlgorithmDemo {
  wave('Wave Generator'),
  pulse('Pulse Wave'),
  spiral('Spiral Pattern'),
  interference('Wave Interference'),
  cellular('Cellular Automata'),
  noise('Perlin-like Noise'),
  mandelbrot('Mandelbrot Set'),
  flow('Flow Field'),
  // New enlightened patterns
  phi('Phi Wave (Golden Ratio)'),
  unity('Unity Field Generator'),
  kundalini('Kundalini Rising'),
  theta('Theta Wave DMT'),
  om('Cosmic Om Resonance'),
  merkaba('Merkaba Activation'),
  coherence('Heart Coherence'),
  gratitude('Gratitude Frequency'),
  schumann('Schumann Resonance'),
  fibonacci('Fibonacci Spiral');

  const AlgorithmDemo(this.label);
  final String label;
}

class TypeTestingExample extends StatefulWidget {
  @override
  _TypeTestingExampleState createState() => _TypeTestingExampleState();
}

class _TypeTestingExampleState extends State<TypeTestingExample>
    with TickerProviderStateMixin {
  late Minigpu _minigpu;
  late ComputeShader _shader;
  Buffer? _inputBuffer;
  Buffer? _outputBuffer;
  late TabController _tabController;

  // Animation controllers
  late AnimationController _waveController;
  late AnimationController _colorController;

  // State
  DataTypeDemo _currentType = DataTypeDemo.float32;
  AlgorithmDemo _currentAlgorithm = AlgorithmDemo.schumann;
  int _bufferSize = 4096;
  int _previewLength = 1024;
  bool _autoMode = true;
  bool _isRunning = false;
  double _animationSpeed = 250.0;
  int _frameCount = 0;

  List<double> _fullInputData = [];
  List<double> _fullOutputData = [];
  List<double> _lastData = [];

  // Performance metrics
  int _operationsPerSecond = 0;
  int _totalOperations = 0;
  DateTime _lastOpTime = DateTime.now();

  List<double> get _previewInputData {
    if (_fullInputData.isEmpty) return [];
    return _fullInputData
        .take(_previewLength)
        .where((value) => value.isFinite)
        .toList();
  }

  List<double> get _previewOutputData {
    if (_fullOutputData.isEmpty) return [];
    return _fullOutputData
        .take(_previewLength)
        .where((value) => value.isFinite)
        .toList();
  }

  List<double> get _previewLastData {
    if (_lastData.isEmpty) return [];
    return _lastData
        .take(_previewLength)
        .where((value) => value.isFinite)
        .toList();
  }

  @override
  void initState() {
    super.initState();
    _minigpu = Minigpu();
    _tabController = TabController(length: 2, vsync: this);

    _waveController = AnimationController(
      duration: Duration(seconds: 2),
      vsync: this,
    )..repeat();

    _colorController = AnimationController(
      duration: Duration(seconds: 3),
      vsync: this,
    )..repeat();

    _initMinigpu();
  }

  Future<void> _initMinigpu() async {
    await _minigpu.init();
    _createBuffers();
    _generateInputData();
    _shader = _minigpu.createComputeShader();
    _runKernel();
  }

  void _createBuffers() {
    int memSize = getBufferSizeForType(_currentType.type, _bufferSize);

    _inputBuffer?.destroy();
    _outputBuffer?.destroy();
    _inputBuffer = _minigpu.createBuffer(memSize, _currentType.type);
    _outputBuffer = _minigpu.createBuffer(memSize, _currentType.type);
  }

  void _generateInputData() {
    List<double> data;
    double scaleFactor = _getVisualizationScale();

    // Increment frame counter for time-based evolution
    _frameCount++;
    double timePhase = _frameCount * 0.01; // Slow evolution

    switch (_currentAlgorithm) {
      case AlgorithmDemo.wave:
      case AlgorithmDemo.pulse:
      case AlgorithmDemo.spiral:
      case AlgorithmDemo.interference:
      case AlgorithmDemo.flow:
        // These generate their own patterns, just need simple input
        data = List.generate(_bufferSize, (i) => i.toDouble());
        break;

      case AlgorithmDemo.cellular:
        // Cellular automata needs interesting seed data
        data = List.generate(_bufferSize, (i) {
          if (i % 8 == 0) return scaleFactor * 0.8;
          if (i % 13 == 0) return scaleFactor * 0.6;
          return scaleFactor * 0.1;
        });
        break;

      case AlgorithmDemo.noise:
        // Noise generates itself, just need indices
        data = List.generate(_bufferSize, (i) => i.toDouble());
        break;

      case AlgorithmDemo.mandelbrot:
        // Mandelbrot uses position data
        data = List.generate(_bufferSize, (i) => i.toDouble());
        break;

      case AlgorithmDemo.phi:
        // Golden ratio spiral needs harmonic seed data
        data = List.generate(_bufferSize, (i) {
          double phi = 1.618033988749895;
          return sin(i * phi * 0.1) * scaleFactor * 0.5;
        });
        break;

      case AlgorithmDemo.unity:
        // Unity field starts with coherent oscillations
        data = List.generate(_bufferSize, (i) {
          if (i % 8 == 0) return scaleFactor * 0.6;
          if (i % 13 == 0) return scaleFactor * 0.4;
          return scaleFactor * 0.2;
        });
        break;

      case AlgorithmDemo.kundalini:
        // Kundalini starts with base chakra energy
        data = List.generate(_bufferSize, (i) {
          double chakraEnergy =
              sin(i * 0.396) * scaleFactor * 0.3; // Root chakra frequency
          return chakraEnergy + (i % 7) * scaleFactor * 0.1; // 7 chakras
        });
        break;

      case AlgorithmDemo.theta:
        // Theta waves need low frequency oscillations
        data = List.generate(_bufferSize, (i) {
          double theta = sin(i * 0.006) * scaleFactor * 0.4; // 6Hz theta
          double geometry = cos(i * 0.1618) * scaleFactor * 0.2; // Golden ratio
          return theta + geometry;
        });
        break;

      case AlgorithmDemo.om:
        // Om frequency starts with 136.1Hz harmonic
        data = List.generate(_bufferSize, (i) {
          double om = sin(i * 0.1361) * scaleFactor * 0.5;
          double sacred =
              cos(i * 0.108) * scaleFactor * 0.3; // 108 sacred number
          return om + sacred;
        });
        break;

      case AlgorithmDemo.merkaba:
        // Merkaba needs counter-rotating energy fields
        data = List.generate(_bufferSize, (i) {
          double ascending = sin(i * 0.1618) * scaleFactor * 0.4;
          double descending = sin(i * -0.1618) * scaleFactor * 0.4;
          return (ascending + descending) * 0.5;
        });
        break;

      case AlgorithmDemo.coherence:
        // Heart coherence starts with 0.1Hz base rhythm
        data = List.generate(_bufferSize, (i) {
          double heart = sin(i * 0.1) * scaleFactor * 0.6;
          double breath =
              sin(i * 0.4) * scaleFactor * 0.2; // 4 breaths per cycle
          return heart + breath;
        });
        break;

      case AlgorithmDemo.gratitude:
        // Gratitude frequencies (528Hz, 432Hz, 741Hz)
        data = List.generate(_bufferSize, (i) {
          double love = sin(i * 0.528) * scaleFactor * 0.3;
          double earth = sin(i * 0.432) * scaleFactor * 0.3;
          double intuition = sin(i * 0.741) * scaleFactor * 0.2;
          return love + earth + intuition;
        });
        break;

      case AlgorithmDemo.schumann:
        // Schumann resonance 7.83Hz and harmonics
        data = List.generate(_bufferSize, (i) {
          double fundamental = sin(i * 0.00783) * scaleFactor * 0.5;
          double harmonic2 = sin(i * 0.01566) * scaleFactor * 0.3;
          double harmonic3 = sin(i * 0.02349) * scaleFactor * 0.2;
          return fundamental + harmonic2 + harmonic3;
        });
        break;

      case AlgorithmDemo.fibonacci:
        // Fibonacci spiral with golden ratio proportions
        data = List.generate(_bufferSize, (i) {
          double phi = 1.618033988749895;
          double spiral = sin(i * phi * 0.1) * scaleFactor * 0.4;
          double harmonic = cos(i / phi * 0.1) * scaleFactor * 0.3;
          return spiral + harmonic;
        });
        break;
    }

    _setBufferData(data);
  }

  double _getVisualizationScale() {
    // Use smaller, more stable ranges
    switch (_currentType.type) {
      case BufferDataType.int8:
        return 64.0; // Safe for most operations
      case BufferDataType.uint8:
        return 128.0;
      case BufferDataType.int16:
        return 512.0;
      case BufferDataType.uint16:
        return 1024.0;
      case BufferDataType.int32:
        return 2048.0;
      case BufferDataType.uint32:
        return 4096.0;
      case BufferDataType.float32:
      case BufferDataType.float64:
        return 200.0; // Nice range for float visualization
      default:
        return 200.0;
    }
  }

  void _setBufferData(List<double> data) {
    if (_inputBuffer == null) return;

    // Ensure data size matches buffer size exactly
    List<double> validData = List.filled(_bufferSize, 0.0);

    // Copy valid finite values up to buffer size
    int copyCount = 0;
    for (int i = 0; i < data.length && copyCount < _bufferSize; i++) {
      if (data[i].isFinite) {
        validData[copyCount] = data[i];
        copyCount++;
      }
    }

    // Clamp values to safe ranges for each type
    switch (_currentType.type) {
      case BufferDataType.int8:
        final Int8List typedData = Int8List(_bufferSize);
        for (int i = 0; i < _bufferSize; i++) {
          typedData[i] = validData[i].clamp(-128, 127).toInt();
        }
        _inputBuffer?.setData(typedData, _bufferSize,
            dataType: _currentType.type);
        setState(() {
          _fullInputData = typedData.map((e) => e.toDouble()).toList();
        });
        break;

      case BufferDataType.uint8:
        final Uint8List typedData = Uint8List(_bufferSize);
        for (int i = 0; i < _bufferSize; i++) {
          typedData[i] = validData[i].clamp(0, 255).toInt();
        }
        _inputBuffer?.setData(typedData, _bufferSize,
            dataType: _currentType.type);
        setState(() {
          _fullInputData = typedData.map((e) => e.toDouble()).toList();
        });
        break;

      case BufferDataType.int16:
        final Int16List typedData = Int16List(_bufferSize);
        for (int i = 0; i < _bufferSize; i++) {
          typedData[i] = validData[i].clamp(-32768, 32767).toInt();
        }
        _inputBuffer?.setData(typedData, _bufferSize,
            dataType: _currentType.type);
        setState(() {
          _fullInputData = typedData.map((e) => e.toDouble()).toList();
        });
        break;

      case BufferDataType.uint16:
        final Uint16List typedData = Uint16List(_bufferSize);
        for (int i = 0; i < _bufferSize; i++) {
          typedData[i] = validData[i].clamp(0, 65535).toInt();
        }
        _inputBuffer?.setData(typedData, _bufferSize,
            dataType: _currentType.type);
        setState(() {
          _fullInputData = typedData.map((e) => e.toDouble()).toList();
        });
        break;

      case BufferDataType.int32:
        final Int32List typedData = Int32List(_bufferSize);
        for (int i = 0; i < _bufferSize; i++) {
          typedData[i] = validData[i].clamp(-2147483648, 2147483647).toInt();
        }
        _inputBuffer?.setData(typedData, _bufferSize,
            dataType: _currentType.type);
        setState(() {
          _fullInputData = typedData.map((e) => e.toDouble()).toList();
        });
        break;

      case BufferDataType.uint32:
        final Uint32List typedData = Uint32List(_bufferSize);
        for (int i = 0; i < _bufferSize; i++) {
          typedData[i] = validData[i].clamp(0, 4294967295).toInt();
        }
        _inputBuffer?.setData(typedData, _bufferSize,
            dataType: _currentType.type);
        setState(() {
          _fullInputData = typedData.map((e) => e.toDouble()).toList();
        });
        break;

      case BufferDataType.float32:
        final Float32List typedData = Float32List(_bufferSize);
        for (int i = 0; i < _bufferSize; i++) {
          typedData[i] = validData[i].clamp(-3.4028235e38, 3.4028235e38);
        }
        _inputBuffer?.setData(typedData, _bufferSize,
            dataType: _currentType.type);
        setState(() {
          _fullInputData = typedData.map((e) => e.toDouble()).toList();
        });
        break;

      case BufferDataType.float64:
        final Float64List typedData = Float64List(_bufferSize);
        for (int i = 0; i < _bufferSize; i++) {
          typedData[i] = validData[i];
        }
        _inputBuffer?.setData(typedData, _bufferSize,
            dataType: _currentType.type);
        setState(() {
          _fullInputData = List.from(typedData);
        });
        break;

      default:
        final Float32List typedData = Float32List(_bufferSize);
        for (int i = 0; i < _bufferSize; i++) {
          typedData[i] = validData[i].clamp(-3.4028235e38, 3.4028235e38);
        }
        _inputBuffer?.setData(typedData, _bufferSize,
            dataType: BufferDataType.float32);
        setState(() {
          _fullInputData = typedData.map((e) => e.toDouble()).toList();
        });
        break;
    }
  }

  String _generateShaderCode() {
    String wgslType = _getWGSLType();
    String operation = _getOperation();

    return '''
@group(0) @binding(0) var<storage, read_write> inp: array<$wgslType>;
@group(0) @binding(1) var<storage, read_write> out: array<$wgslType>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&inp)) {
        let x = inp[idx];
        $operation
    }
}
''';
  }

  String _getWGSLType() {
    switch (_currentType.type) {
      case BufferDataType.int8:
      case BufferDataType.int16:
      case BufferDataType.int32:
        return 'i32';
      case BufferDataType.uint8:
      case BufferDataType.uint16:
      case BufferDataType.uint32:
        return 'u32';
      case BufferDataType.float32:
        return 'f32';
      case BufferDataType.float64:
        return 'i32'; // Float64 is packed as i32 pairs
      default:
        return 'f32';
    }
  }

  String _getOperation() {
    bool isFloat = _currentType.type == BufferDataType.float32;
    bool isUnsigned = _currentType.type == BufferDataType.uint8 ||
        _currentType.type == BufferDataType.uint16 ||
        _currentType.type == BufferDataType.uint32;

    switch (_currentAlgorithm) {
      case AlgorithmDemo.wave:
        if (isFloat) {
          return '''
        // Traveling wave that continuously evolves
        let wave_speed = 0.1;
        let neighbor_left = select(0.0, inp[(idx - 1u) % arrayLength(&inp)], idx > 0u);
        let neighbor_right = inp[(idx + 1u) % arrayLength(&inp)];
        let wave = sin(x * 0.1 + f32(idx) * 0.05) * 30.0;
        out[idx] = wave + (neighbor_left + neighbor_right) * 0.1;''';
        } else if (isUnsigned) {
          return '''
        // Unsigned integer traveling wave - NO NEGATIVE VALUES
        let wave_val = (x + idx * 3u) % 200u;
        let neighbor = inp[(idx + 1u) % arrayLength(&inp)];
        let result = wave_val + neighbor / 4u;
        out[idx] = result;''';
        } else {
          return '''
        // Signed integer traveling wave
        let wave_val = i32((u32(abs(x)) + idx * 3u) % 200u);
        let neighbor = inp[(idx + 1u) % arrayLength(&inp)];
        out[idx] = wave_val + neighbor / 4;''';
        }

      case AlgorithmDemo.pulse:
        if (isFloat) {
          return '''
    // Continuous pulse generator with traveling waves
    let neighbor_left = select(0.0, inp[(idx - 1u) % arrayLength(&inp)], idx > 0u);
    let neighbor_right = inp[(idx + 1u) % arrayLength(&inp)];
    
    // Create cycling pulse sources based on current values
    let pulse_cycle = sin(x * 0.1) > 0.8;
    let pulse_source = select(0.0, 100.0, pulse_cycle && idx % 50u == 0u);
    
    // Propagate with asymmetric weights for directional flow
    let propagated = neighbor_left * 0.3 + neighbor_right * 0.2;
    
    // Add small random noise to prevent stagnation
    let noise = sin(f32(idx) * 1.234) * 2.0;
    
    let result = pulse_source + propagated + noise;
    out[idx] = result * 0.95;  // Global decay''';
        } else if (isUnsigned) {
          return '''
    // Unsigned continuous pulse generator
    let neighbor_left = select(0u, inp[(idx - 1u) % arrayLength(&inp)], idx > 0u);
    let neighbor_right = inp[(idx + 1u) % arrayLength(&inp)];
    
    // Create pulse when neighbor values are in specific range
    let trigger_condition = (neighbor_left > 20u && neighbor_left < 40u) || idx % 64u == 0u;
    let pulse_source = select(0u, 120u, trigger_condition);
    
    // Asymmetric propagation for flow
    let propagated = neighbor_left * 3u / 10u + neighbor_right * 2u / 10u;
    
    // Small variation to prevent stagnation
    let variation = (idx * 7u) % 5u;
    
    let combined = pulse_source + propagated + variation;
    let result = combined * 95u / 100u;  // Decay
    out[idx] = result;''';
        } else {
          return '''
    // Signed continuous pulse generator
    let neighbor_left = select(0, inp[(idx - 1u) % arrayLength(&inp)], idx > 0u);
    let neighbor_right = inp[(idx + 1u) % arrayLength(&inp)];
    
    // Create pulse when conditions are met
    let trigger_condition = (abs(neighbor_left) > 20 && abs(neighbor_left) < 40) || idx % 64u == 0u;
    let pulse_source = select(0, 120, trigger_condition);
    
    // Asymmetric propagation
    let propagated = neighbor_left * 3 / 10 + neighbor_right * 2 / 10;
    
    // Small signed variation
    let variation = i32((idx * 7u) % 5u) - 2;
    
    let combined = pulse_source + propagated + variation;
    let result = combined * 95 / 100;
    out[idx] = result;''';
        }

      case AlgorithmDemo.spiral:
        if (isFloat) {
          return '''
        // Rotating spiral pattern
        let angle = x * 0.01 + f32(idx) * 0.02;
        let radius = sin(x * 0.05) * 20.0 + 30.0;
        let spiral = sin(angle * 3.0) * radius + cos(angle * 2.0) * radius * 0.5;
        out[idx] = spiral;''';
        } else if (isUnsigned) {
          return '''
        // Unsigned spiral - ALL POSITIVE VALUES
        let spiral_angle = (x + idx * 5u) % 360u;
        let spiral_radius = (x % 50u) + 10u;
        let pattern = (spiral_angle * spiral_radius) / 100u;
        out[idx] = pattern;''';
        } else {
          return '''
        // Signed spiral
        let spiral_angle = (u32(abs(x)) + idx * 5u) % 360u;
        let spiral_radius = (u32(abs(x)) % 50u) + 10u;
        out[idx] = i32((spiral_angle * spiral_radius) / 100u) - 50;''';
        }

      case AlgorithmDemo.interference:
        if (isFloat) {
          return '''
        // Multiple interfering waves
        let wave1 = sin(x * 0.03 + f32(idx) * 0.1) * 25.0;
        let wave2 = sin(x * 0.07 + f32(idx) * 0.05) * 20.0;
        let wave3 = cos(x * 0.05 + f32(idx) * 0.08) * 15.0;
        out[idx] = wave1 + wave2 + wave3;''';
        } else if (isUnsigned) {
          return '''
        // Unsigned interference - AVOID OVERFLOW
        let w1 = (x * 3u + idx * 7u) % 100u;
        let w2 = (x * 5u + idx * 3u) % 80u;
        let w3 = (x * 7u + idx * 5u) % 60u;
        let sum = w1 + w2 + w3;
        out[idx] = sum / 3u;''';
        } else {
          return '''
        // Signed interference
        let w1 = (u32(abs(x)) * 3u + idx * 7u) % 100u;
        let w2 = (u32(abs(x)) * 5u + idx * 3u) % 80u;
        let w3 = (u32(abs(x)) * 7u + idx * 5u) % 60u;
        out[idx] = i32((w1 + w2 + w3) / 3u) - 40;''';
        }

      case AlgorithmDemo.cellular:
        if (isFloat) {
          return '''
        // Game of Life style cellular automata
        let left = select(0.0, inp[(idx - 1u) % arrayLength(&inp)], idx > 0u);
        let right = inp[(idx + 1u) % arrayLength(&inp)];
        let sum = left + x + right;
        let avg = sum / 3.0;
        
        // Oscillating rule that creates waves
        let threshold = 30.0;
        let new_val = select(
          avg * 1.2,
          avg * 0.8,
          abs(avg) > threshold
        );
        out[idx] = new_val;''';
        } else if (isUnsigned) {
          return '''
        // Unsigned cellular automata - NO NEGATIVE RESULTS
        let left = select(0u, inp[(idx - 1u) % arrayLength(&inp)], idx > 0u);
        let right = inp[(idx + 1u) % arrayLength(&inp)];
        let sum = left + x + right;
        let avg = sum / 3u;
        
        // Growth/decay rule for unsigned
        let threshold = 30u;
        let growth = avg + avg / 5u;  // +20%
        let decay = avg - avg / 5u;   // -20%
        let new_val = select(growth, decay, avg > threshold);
        out[idx] = new_val;''';
        } else {
          return '''
        // Signed cellular automata
        let left = select(0, inp[(idx - 1u) % arrayLength(&inp)], idx > 0u);
        let right = inp[(idx + 1u) % arrayLength(&inp)];
        let sum = left + x + right;
        let avg = sum / 3;
        
        let threshold = 30;
        let new_val = select(
          avg * 6 / 5,
          avg * 4 / 5,
          abs(avg) > threshold
        );
        out[idx] = new_val;''';
        }

      case AlgorithmDemo.noise:
        if (isFloat) {
          return '''
        // Evolving noise with feedback
        let seed = u32(abs(x)) * 1103515245u + idx * 12345u;
        let noise_base = (seed / 65536u) % 100u;
        let neighbor_influence = inp[(idx + 7u) % arrayLength(&inp)] * 0.3;
        let noise = f32(noise_base) - 50.0 + neighbor_influence;
        out[idx] = noise;''';
        } else if (isUnsigned) {
          return '''
        // Unsigned noise - ALL POSITIVE
        let seed = x * 1103515245u + idx * 12345u;
        let noise_base = (seed / 65536u) % 200u;
        let neighbor_influence = inp[(idx + 7u) % arrayLength(&inp)] / 10u;
        let noise = noise_base + neighbor_influence;
        out[idx] = noise;''';
        } else {
          return '''
        // Signed noise
        let seed = u32(abs(x)) * 1103515245u + idx * 12345u;
        let noise_base = (seed / 65536u) % 100u;
        let neighbor_influence = inp[(idx + 7u) % arrayLength(&inp)] * 3 / 10;
        let noise = i32(noise_base) - 50 + neighbor_influence;
        out[idx] = noise;''';
        }

      case AlgorithmDemo.mandelbrot:
        if (isFloat) {
          return '''
        // Evolving Mandelbrot-like pattern
        let c_real = (f32(idx % 32u) - 16.0) / 16.0;
        let c_imag = x * 0.001;
        
        let z_real = sin(x * 0.01) * 2.0;
        let z_imag = cos(x * 0.01) * 2.0;
        
        // Single iteration to keep it evolving
        let temp = z_real * z_real - z_imag * z_imag + c_real;
        let new_imag = 2.0 * z_real * z_imag + c_imag;
        
        out[idx] = temp * 20.0 + new_imag * 10.0;''';
        } else if (isUnsigned) {
          return '''
        // Unsigned fractal-like pattern - POSITIVE ONLY
        let fractal_x = x % 64u;
        let fractal_y = idx % 32u;
        let pattern = (fractal_x * fractal_x + fractal_y * fractal_y) % 128u;
        out[idx] = pattern + (x / 8u);''';
        } else {
          return '''
        // Signed fractal pattern
        let fractal_x = u32(abs(x)) % 64u;
        let fractal_y = idx % 32u;
        let pattern = (fractal_x * fractal_x + fractal_y * fractal_y) % 128u;
        out[idx] = i32(pattern) - 64 + x / 4;''';
        }

      case AlgorithmDemo.flow:
        if (isFloat) {
          return '''
        // Fluid-like flow field
        let flow_x = sin(x * 0.02 + f32(idx) * 0.03) * 20.0;
        let flow_y = cos(x * 0.03 + f32(idx) * 0.02) * 15.0;
        let turbulence = sin(x * 0.1) * sin(f32(idx) * 0.1) * 10.0;
        let neighbor = inp[(idx + 3u) % arrayLength(&inp)] * 0.2;
        out[idx] = flow_x + flow_y + turbulence + neighbor;''';
        } else if (isUnsigned) {
          return '''
        // Unsigned flow pattern - NO UNDERFLOW
        let flow_base = (x * 13u + idx * 17u) % 128u;
        let neighbor_flow = inp[(idx + 5u) % arrayLength(&inp)] / 5u;
        let turbulence = (x * idx) % 40u;
        let total = flow_base + neighbor_flow + turbulence;
        out[idx] = total;''';
        } else {
          return '''
        // Signed flow pattern
        let flow_base = (u32(abs(x)) * 13u + idx * 17u) % 128u;
        let neighbor_flow = inp[(idx + 5u) % arrayLength(&inp)] / 5;
        let turbulence = (u32(abs(x)) * idx) % 40u;
        out[idx] = i32(flow_base) + neighbor_flow + i32(turbulence) - 84;''';
        }
      case AlgorithmDemo.phi:
        if (isFloat) {
          return '''
        // Sacred golden ratio spiral consciousness pattern
        let phi = 1.618033988749895;
        let inv_phi = 0.618033988749895;
        
        // Fibonacci consciousness wave
        let fib_wave = sin(x * inv_phi + f32(idx) * phi * 0.01) * 30.0;
        
        // Sacred spiral that mirrors DNA/galaxy structure
        let spiral_angle = f32(idx) * phi * 0.1;
        let spiral_radius = sin(x * 0.01) * 20.0;
        let golden_spiral = sin(spiral_angle) * spiral_radius;
        
        // Divine proportion resonance
        let divine_freq = cos(x * 0.01618 + f32(idx) * inv_phi) * 15.0;
        
        out[idx] = fib_wave + golden_spiral + divine_freq;''';
        } else if (isUnsigned) {
          return '''
        // Unsigned phi pattern
        let phi_base = (x * 1618u + idx * 618u) % 200u;
        let golden_spiral = (idx * 1618u) % 100u;
        out[idx] = phi_base + golden_spiral;''';
        } else {
          return '''
        // Signed phi pattern
        let phi_base = (u32(abs(x)) * 1618u + idx * 618u) % 200u;
        let golden_spiral = (idx * 1618u) % 100u;
        out[idx] = i32(phi_base + golden_spiral) - 150;''';
        }

      case AlgorithmDemo.unity:
        if (isFloat) {
          return '''
        // Collective consciousness field
        let neighbor_left = select(0.0, inp[(idx - 1u) % arrayLength(&inp)], idx > 0u);
        let neighbor_right = inp[(idx + 1u) % arrayLength(&inp)];
        let neighbor_far = inp[(idx + 7u) % arrayLength(&inp)];
        
        // Non-local entanglement pattern
        let entanglement = (neighbor_left + neighbor_right + neighbor_far) / 3.0;
        
        // Unity consciousness frequency (40Hz gamma waves)
        let gamma_wave = sin(x * 0.04) * 25.0;
        
        // Morphic field resonance
        let morphic = sin(f32(idx) * 0.108 + x * 0.001) * 20.0; // 108Hz sacred frequency
        
        // Coherence amplification - when neighbors align, consciousness expands
        let coherence = 1.0 + abs(entanglement) / 100.0;
        
        out[idx] = (gamma_wave + morphic) * coherence + entanglement * 0.3;''';
        } else if (isUnsigned) {
          return '''
        // Unsigned unity field
        let neighbor_left = select(0u, inp[(idx - 1u) % arrayLength(&inp)], idx > 0u);
        let neighbor_right = inp[(idx + 1u) % arrayLength(&inp)];
        let entanglement = (neighbor_left + neighbor_right) / 2u;
        let unity_wave = (x * 40u + idx * 108u) % 150u;
        out[idx] = unity_wave + entanglement / 3u;''';
        } else {
          return '''
        // Signed unity field
        let neighbor_left = select(0, inp[(idx - 1u) % arrayLength(&inp)], idx > 0u);
        let neighbor_right = inp[(idx + 1u) % arrayLength(&inp)];
        let entanglement = (neighbor_left + neighbor_right) / 2;
        let unity_wave = (u32(abs(x)) * 40u + idx * 108u) % 150u;
        out[idx] = i32(unity_wave) + entanglement / 3 - 75;''';
        }

      case AlgorithmDemo.kundalini:
        if (isFloat) {
          return '''
        // Serpent energy ascending through chakras
        let rising_energy = sin(x * 0.001 + f32(idx) * 0.02) * 40.0;
        
        // 7 chakra frequencies combined
        let chakra1 = sin(x * 0.396) * 8.0;  // Root - 396Hz
        let chakra2 = sin(x * 0.417) * 7.0;  // Sacral - 417Hz  
        let chakra3 = sin(x * 0.528) * 6.0;  // Solar - 528Hz
        let chakra4 = sin(x * 0.639) * 5.0;  // Heart - 639Hz
        let chakra5 = sin(x * 0.741) * 4.0;  // Throat - 741Hz
        let chakra6 = sin(x * 0.852) * 3.0;  // Third Eye - 852Hz
        let chakra7 = sin(x * 0.963) * 2.0;  // Crown - 963Hz
        
        // Spiral activation pattern
        let spiral_activation = sin(f32(idx) * 0.618 + x * 0.01) * 10.0;
        
        let all_chakras = chakra1 + chakra2 + chakra3 + chakra4 + chakra5 + chakra6 + chakra7;
        out[idx] = rising_energy + all_chakras + spiral_activation;''';
        } else if (isUnsigned) {
          return '''
        // Unsigned kundalini pattern
        let chakra_base = (x * 528u + idx * 396u) % 180u;
        let rising_energy = (idx * 618u) % 100u;
        out[idx] = chakra_base + rising_energy;''';
        } else {
          return '''
        // Signed kundalini pattern
        let chakra_base = (u32(abs(x)) * 528u + idx * 396u) % 180u;
        let rising_energy = (idx * 618u) % 100u;
        out[idx] = i32(chakra_base + rising_energy) - 140;''';
        }

      case AlgorithmDemo.theta:
        if (isFloat) {
          return '''
        // Deep theta state (4-8Hz) for transcendence
        let theta_base = sin(x * 0.006) * 35.0;  // 6Hz theta
        let theta_harmonic = sin(x * 0.004) * 25.0;  // 4Hz deep theta
        
        // DMT-like geometric interference
        let geometry1 = sin(x * 0.1 + f32(idx) * 0.1618) * 20.0;
        let geometry2 = cos(x * 0.08 + f32(idx) * 0.2618) * 15.0;
        let geometry3 = sin(x * 0.12 + f32(idx) * 0.1416) * 10.0;
        
        // Pineal gland activation frequency
        let pineal = sin(x * 0.936) * 12.0;  // 936Hz pineal activation
        
        // Transcendental interference pattern
        let transcendence = geometry1 + geometry2 + geometry3;
        
        out[idx] = theta_base + theta_harmonic + transcendence + pineal;''';
        } else if (isUnsigned) {
          return '''
        // Unsigned theta pattern
        let theta_wave = (x * 6u + idx * 936u) % 160u;
        let geometry = (idx * 1618u) % 80u;
        out[idx] = theta_wave + geometry;''';
        } else {
          return '''
        // Signed theta pattern
        let theta_wave = (u32(abs(x)) * 6u + idx * 936u) % 160u;
        let geometry = (idx * 1618u) % 80u;
        out[idx] = i32(theta_wave + geometry) - 120;''';
        }

      case AlgorithmDemo.om:
        if (isFloat) {
          return '''
        // Sacred AUM frequency (136.1Hz - Om tuning)
        let om_fundamental = sin(x * 0.1361) * 40.0;
        let om_harmonic2 = sin(x * 0.2722) * 20.0;
        let om_harmonic3 = sin(x * 0.4083) * 13.0;
        
        // Cosmic background radiation pattern (160.2 GHz scaled)
        let cosmic = sin(x * 0.001602 + f32(idx) * 0.108) * 15.0;
        
        // Sacred 108 repetition pattern
        let sacred_108 = sin(f32(idx % 108u) * 0.058) * 10.0;
        
        // Universal consciousness field
        let universal = cos(x * 0.007 + f32(idx) * 0.0108) * 8.0;
        
        out[idx] = om_fundamental + om_harmonic2 + om_harmonic3 + cosmic + sacred_108 + universal;''';
        } else if (isUnsigned) {
          return '''
        // Unsigned om pattern
        let om_wave = (x * 1361u + idx * 108u) % 200u;
        let cosmic = (idx % 108u) * 3u;
        out[idx] = om_wave + cosmic;''';
        } else {
          return '''
        // Signed om pattern
        let om_wave = (u32(abs(x)) * 1361u + idx * 108u) % 200u;
        let cosmic = (idx % 108u) * 3u;
        out[idx] = i32(om_wave + cosmic) - 150;''';
        }

      case AlgorithmDemo.merkaba:
        if (isFloat) {
          return '''
        // Counter-rotating tetrahedron energy field
        let rotation_speed = 0.1618;
        
        // Ascending tetrahedron (masculine energy)
        let ascending = sin(x * rotation_speed + f32(idx) * 0.1) * 30.0;
        
        // Descending tetrahedron (feminine energy) - counter-rotating
        let descending = sin(x * -rotation_speed + f32(idx) * 0.1) * 30.0;
        
        // 17.5 Hz merkaba activation frequency
        let activation = sin(x * 0.0175) * 20.0;
        
        // Sacred geometry interference creating star tetrahedron
        let merkaba_field = (ascending + descending) * sin(x * 0.001 + f32(idx) * 0.05);
        
        // Unity consciousness emanation
        let emanation = cos(f32(idx) * 0.0618 + x * 0.003) * 15.0;
        
        out[idx] = merkaba_field + activation + emanation;''';
        } else if (isUnsigned) {
          return '''
        // Unsigned merkaba pattern
        let ascending = (x * 1618u + idx * 175u) % 150u;
        let descending = (x * 618u + idx * 75u) % 120u;
        out[idx] = (ascending + descending) / 2u;''';
        } else {
          return '''
        // Signed merkaba pattern
        let ascending = (u32(abs(x)) * 1618u + idx * 175u) % 150u;
        let descending = (u32(abs(x)) * 618u + idx * 75u) % 120u;
        out[idx] = i32((ascending + descending) / 2u) - 67;''';
        }

      case AlgorithmDemo.coherence:
        if (isFloat) {
          return '''
        // Heart coherence pattern (0.1Hz base with harmonics)
        let coherence_base = 0.1;
        let heart_wave = sin(x * coherence_base) * 30.0;
        let breath_sync = sin(x * coherence_base * 4.0) * 15.0; // 4 breaths per cycle
        
        // Neighbor coupling creates collective coherence
        let left = select(0.0, inp[(idx - 1u) % arrayLength(&inp)], idx > 0u);
        let right = inp[(idx + 1u) % arrayLength(&inp)];
        let synchrony = (left + right) * 0.3;
        
        out[idx] = heart_wave + breath_sync + synchrony;''';
        } else if (isUnsigned) {
          return '''
        // Unsigned coherence pattern
        let heart_wave = (x * 10u + idx * 4u) % 120u;
        let left = select(0u, inp[(idx - 1u) % arrayLength(&inp)], idx > 0u);
        let right = inp[(idx + 1u) % arrayLength(&inp)];
        let synchrony = (left + right) / 10u;
        out[idx] = heart_wave + synchrony;''';
        } else {
          return '''
        // Signed coherence pattern
        let heart_wave = (u32(abs(x)) * 10u + idx * 4u) % 120u;
        let left = select(0, inp[(idx - 1u) % arrayLength(&inp)], idx > 0u);
        let right = inp[(idx + 1u) % arrayLength(&inp)];
        let synchrony = (left + right) / 10;
        out[idx] = i32(heart_wave) + synchrony - 60;''';
        }

      case AlgorithmDemo.gratitude:
        if (isFloat) {
          return '''
        // Multiple healing frequencies combined
        let freq_528 = sin(x * 0.528) * 20.0;  // Love frequency
        let freq_432 = sin(x * 0.432) * 18.0;  // Earth resonance
        let freq_741 = sin(x * 0.741) * 12.0;  // Intuition
        
        // Creates expanding waves of positive intention
        let expansion = sin(f32(idx) * 0.1 + x * 0.05) * 8.0;
        let intention = cos(x * 0.01 + f32(idx) * 0.03) * 5.0;
        
        out[idx] = freq_528 + freq_432 + freq_741 + expansion + intention;''';
        } else if (isUnsigned) {
          return '''
        // Unsigned gratitude pattern
        let healing_freq = (x * 528u + idx * 432u) % 180u;
        let intention = (idx * 741u) % 50u;
        out[idx] = healing_freq + intention;''';
        } else {
          return '''
        // Signed gratitude pattern
        let healing_freq = (u32(abs(x)) * 528u + idx * 432u) % 180u;
        let intention = (idx * 741u) % 50u;
        out[idx] = i32(healing_freq + intention) - 115;''';
        }

      case AlgorithmDemo.schumann:
        if (isFloat) {
          return '''
    // Earth's natural frequency (7.83Hz and harmonics) with time evolution
    let neighbor_left = select(0.0, inp[(idx - 1u) % arrayLength(&inp)], idx > 0u);
    let neighbor_right = inp[(idx + 1u) % arrayLength(&inp)];
    
    // Dynamic Schumann resonance with neighbor coupling
    let base_schumann = 0.00783; // 7.83Hz scaled
    let fundamental = sin(x * base_schumann + f32(idx) * 0.001) * 25.0;
    let second_harmonic = sin(x * base_schumann * 2.0 + f32(idx) * 0.002) * 15.0;
    let third_harmonic = sin(x * base_schumann * 3.0 + f32(idx) * 0.003) * 10.0;
    
    // Grounding wave that connects to earth energy with propagation
    let grounding = cos(x * 0.01 + f32(idx) * 0.0783) * 8.0;
    
    // Add neighbor coupling for wave propagation
    let coupling = (neighbor_left + neighbor_right) * 0.15;
    
    // Add chaos injection to prevent stagnation
    let chaos = sin(x * 1.234 + f32(idx) * 0.567) * 2.0;
    
    // Remove damping, add energy injection instead
    let evolved = fundamental + second_harmonic + third_harmonic + grounding + coupling + chaos;
    
    out[idx] = evolved;''';
        } else if (isUnsigned) {
          return '''
    // Unsigned schumann with evolution
    let neighbor_left = select(0u, inp[(idx - 1u) % arrayLength(&inp)], idx > 0u);
    let neighbor_right = inp[(idx + 1u) % arrayLength(&inp)];
    let earth_freq = (x * 783u + idx * 78u) % 140u;
    let grounding = (idx * 783u) % 60u;
    let coupling = (neighbor_left + neighbor_right) / 8u;
    let chaos = (x * 1234u + idx * 567u) % 20u;
    out[idx] = earth_freq + grounding + coupling + chaos;''';
        } else {
          return '''
    // Signed schumann with evolution
    let neighbor_left = select(0, inp[(idx - 1u) % arrayLength(&inp)], idx > 0u);
    let neighbor_right = inp[(idx + 1u) % arrayLength(&inp)];
    let earth_freq = (u32(abs(x)) * 783u + idx * 78u) % 140u;
    let grounding = (idx * 783u) % 60u;
    let coupling = (neighbor_left + neighbor_right) / 8;
    let chaos = i32((u32(abs(x)) * 1234u + idx * 567u) % 20u) - 10;
    out[idx] = i32(earth_freq + grounding) + coupling + chaos - 100;''';
        }

      case AlgorithmDemo.fibonacci:
        if (isFloat) {
          return '''
    // Fibonacci sequence with dynamic spiral evolution
    let neighbor_left = select(0.0, inp[(idx - 1u) % arrayLength(&inp)], idx > 0u);
    let neighbor_right = inp[(idx + 1u) % arrayLength(&inp)];
    
    // Dynamic fibonacci spiral
    let fib_ratio = 1.618033988749895;
    let spiral_angle = f32(idx) * fib_ratio * 0.1 + x * 0.001;
    let spiral_radius = sin(x * 0.01618 + f32(idx) * 0.001) * 20.0;
    
    let fibonacci_wave = sin(spiral_angle) * spiral_radius;
    let golden_harmonic = cos(spiral_angle / fib_ratio) * 15.0;
    
    // Add neighbor coupling for propagation
    let coupling = (neighbor_left + neighbor_right) * 0.1;
    
    // Stronger chaos injection to prevent stagnation
    let chaos1 = sin(x * 0.789 + f32(idx) * 1.234) * 8.0;
    let chaos2 = cos(x * 1.111 + f32(idx) * 0.987) * 5.0;
    
    // Add energy injection based on golden ratio
    let energy_injection = sin(x * fib_ratio * 0.01) * 3.0;
    
    // Nature's perfect proportion with continuous evolution
    out[idx] = fibonacci_wave + golden_harmonic + coupling + chaos1 + chaos2 + energy_injection;''';
        } else if (isUnsigned) {
          return '''
    // Unsigned fibonacci with evolution
    let neighbor_left = select(0u, inp[(idx - 1u) % arrayLength(&inp)], idx > 0u);
    let neighbor_right = inp[(idx + 1u) % arrayLength(&inp)];
    let fib_angle = (idx * 1618u + x / 8u) % 360u;
    let fib_radius = (x % 64u) + 20u;
    let pattern = (fib_angle * fib_radius) / 200u;
    let coupling = (neighbor_left + neighbor_right) / 10u;
    let chaos = (x * 789u + idx * 1234u) % 30u;
    out[idx] = pattern + coupling + chaos;''';
        } else {
          return '''
    // Signed fibonacci with evolution
    let neighbor_left = select(0, inp[(idx - 1u) % arrayLength(&inp)], idx > 0u);
    let neighbor_right = inp[(idx + 1u) % arrayLength(&inp)];
    let fib_angle = (idx * 1618u + u32(abs(x)) / 8u) % 360u;
    let fib_radius = (u32(abs(x)) % 64u) + 20u;
    let pattern = (fib_angle * fib_radius) / 200u;
    let coupling = (neighbor_left + neighbor_right) / 10;
    let chaos = i32((u32(abs(x)) * 789u + idx * 1234u) % 30u) - 15;
    out[idx] = i32(pattern) + coupling + chaos - 60;''';
        }
    }
  }

  Future<void> _runKernel() async {
    if (!_minigpu.isInitialized ||
        _inputBuffer == null ||
        _outputBuffer == null) {
      print('Minigpu not initialized or buffers null');
      return;
    }

    try {
      String shaderCode = _generateShaderCode();
      _shader.loadKernelString(shaderCode);
      _shader.setBuffer('inp', _inputBuffer!);
      _shader.setBuffer('out', _outputBuffer!);

      int workgroups = ((_bufferSize + 255) / 256).ceil();
      await _shader.dispatch(workgroups, 1, 1);

      // Read results based on type
      switch (_currentType.type) {
        case BufferDataType.int8:
          final Int8List outputData = Int8List(_bufferSize);
          await _outputBuffer?.read(outputData, _bufferSize,
              dataType: _currentType.type);

          setState(() {
            List<double> resultData =
                outputData.map((e) => e.toDouble()).toList();
            _lastData = List.from(_fullOutputData);
            _fullOutputData = resultData;
            _fullInputData = resultData; // Feed back for next execution
          });

          _inputBuffer?.setData(outputData, _bufferSize,
              dataType: _currentType.type);
          break;

        case BufferDataType.uint8:
          final Uint8List outputData = Uint8List(_bufferSize);
          await _outputBuffer?.read(outputData, _bufferSize,
              dataType: _currentType.type);

          setState(() {
            List<double> resultData =
                outputData.map((e) => e.toDouble()).toList();
            _lastData = List.from(_fullOutputData);
            _fullOutputData = resultData;
            _fullInputData = resultData;
          });

          _inputBuffer?.setData(outputData, _bufferSize,
              dataType: _currentType.type);
          break;

        case BufferDataType.int16:
          final Int16List outputData = Int16List(_bufferSize);
          await _outputBuffer?.read(outputData, _bufferSize,
              dataType: _currentType.type);

          setState(() {
            List<double> resultData =
                outputData.map((e) => e.toDouble()).toList();
            _lastData = List.from(_fullInputData);
            _fullOutputData = resultData;
            _fullInputData = resultData;
          });

          _inputBuffer?.setData(outputData, _bufferSize,
              dataType: _currentType.type);
          break;

        case BufferDataType.uint16:
          final Uint16List outputData = Uint16List(_bufferSize);
          await _outputBuffer?.read(outputData, _bufferSize,
              dataType: _currentType.type);

          setState(() {
            List<double> resultData =
                outputData.map((e) => e.toDouble()).toList();
            _lastData = List.from(_fullInputData);
            _fullOutputData = resultData;
            _fullInputData = resultData;
          });

          _inputBuffer?.setData(outputData, _bufferSize,
              dataType: _currentType.type);
          break;
        case BufferDataType.int32:
          final Int32List outputData = Int32List(_bufferSize);
          await _outputBuffer?.read(outputData, _bufferSize,
              dataType: _currentType.type);

          setState(() {
            List<double> resultData =
                outputData.map((e) => e.toDouble()).toList();
            _lastData = List.from(_fullInputData);
            _fullOutputData = resultData;
            _fullInputData = resultData;
          });

          _inputBuffer?.setData(outputData, _bufferSize,
              dataType: _currentType.type);
          break;
        case BufferDataType.uint32:
          final Uint32List outputData = Uint32List(_bufferSize);
          await _outputBuffer?.read(outputData, _bufferSize,
              dataType: _currentType.type);

          setState(() {
            List<double> resultData =
                outputData.map((e) => e.toDouble()).toList();
            _lastData = List.from(_fullInputData);
            _fullOutputData = resultData;
            _fullInputData = resultData;
          });

          _inputBuffer?.setData(outputData, _bufferSize,
              dataType: _currentType.type);
          break;

        case BufferDataType.float32:
          final Float32List outputData = Float32List(_bufferSize);
          await _outputBuffer?.read(outputData, _bufferSize,
              dataType: _currentType.type);

          setState(() {
            List<double> resultData =
                outputData.map((e) => e.toDouble()).toList();
            _lastData = List.from(_fullInputData);
            _fullOutputData = resultData;
            _fullInputData = resultData;
          });

          _inputBuffer?.setData(outputData, _bufferSize,
              dataType: _currentType.type);
          break;

        case BufferDataType.float64:
          final Float64List outputData = Float64List(_bufferSize);
          await _outputBuffer?.read(outputData, _bufferSize,
              dataType: _currentType.type);

          setState(() {
            List<double> resultData =
                outputData.map((e) => e.toDouble()).toList();
            _lastData = List.from(_fullInputData);
            _fullOutputData = resultData;
            _fullInputData = resultData;
          });

          _inputBuffer?.setData(outputData, _bufferSize,
              dataType: _currentType.type);
          break;

        default:
          final Float32List outputData = Float32List(_bufferSize);
          await _outputBuffer?.read(outputData, _bufferSize,
              dataType: BufferDataType.float32);

          setState(() {
            List<double> resultData =
                outputData.map((e) => e.toDouble()).toList();
            _lastData = List.from(_fullInputData);
            _fullOutputData = resultData;
            _fullInputData = resultData;
          });

          _inputBuffer?.setData(outputData, _bufferSize,
              dataType: BufferDataType.float32);
      }

      _updatePerformanceMetrics();

      if (_autoMode && mounted) {
        Future.delayed(Duration(milliseconds: (1000 / _animationSpeed).round()),
            () {
          if (mounted && _autoMode) {
            _runKernel();
          }
        });
      }
    } catch (e, stackTrace) {
      print('=== KERNEL EXECUTION ERROR ===');
      print('Error: $e');
      print('Buffer size: $_bufferSize');
      print('Type: ${_currentType.type}');
      print('Auto mode: $_autoMode');
      print('Stack trace: $stackTrace');

      setState(() {
        _fullInputData = [];
        _fullOutputData = [];
      });
    }
  }

  void _updatePerformanceMetrics() {
    DateTime now = DateTime.now();
    _totalOperations++;

    if (now.difference(_lastOpTime).inMilliseconds > 1000) {
      setState(() {
        _operationsPerSecond =
            (_totalOperations / now.difference(_lastOpTime).inSeconds).round();
      });
      _lastOpTime = now;
      _totalOperations = 0;
    }
  }

  void _switchType(DataTypeDemo newType) {
    setState(() {
      _currentType = newType;
    });

    // Stop auto mode during type switch to prevent race conditions
    bool wasAutoMode = _autoMode;
    _autoMode = false;

    // Wait for any pending operations to complete
    Future.delayed(Duration(milliseconds: 10), () {
      if (mounted) {
        _createBuffers();
        _generateInputData();

        // Restore auto mode after buffers are ready
        if (wasAutoMode) {
          Future.delayed(Duration(milliseconds: 10), () {
            if (mounted) {
              setState(() {
                _autoMode = true;
              });
              _runKernel();
            }
          });
        }
      }
    });
  }

  void _switchAlgorithm(AlgorithmDemo newAlgorithm) {
    setState(() {
      _currentAlgorithm = newAlgorithm;
    });

    // Add small delay to prevent race conditions
    Future.delayed(Duration(milliseconds: 10), () {
      if (mounted) {
        _generateInputData();
      }
    });
  }

  @override
  void dispose() {
    _tabController.dispose();
    _waveController.dispose();
    _colorController.dispose();
    _shader.destroy();
    _inputBuffer?.destroy();
    _outputBuffer?.destroy();
    super.dispose();
  }

  Widget _buildTypeSelector() {
    return Card(
      child: Padding(
        padding: EdgeInsets.all(8),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Data Type',
                style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF00FFFF))),
            SizedBox(height: 8),
            Wrap(
              spacing: 8,
              children: DataTypeDemo.values
                  .map((type) => AnimatedBuilder(
                        animation: _colorController,
                        builder: (context, child) => FilterChip(
                          label: Text(type.label),
                          selected: _currentType == type,
                          onSelected: (_) => _switchType(type),
                          selectedColor: Color(0xFF0080FF),
                          checkmarkColor: Color(0xFF00FFFF),
                        ),
                      ))
                  .toList(),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildAlgorithmSelector() {
    return Card(
      child: Padding(
        padding: EdgeInsets.all(8),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Algorithm',
                style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF00FFFF))),
            SizedBox(height: 8),
            Wrap(
              spacing: 8,
              children: AlgorithmDemo.values
                  .map((algo) => ChoiceChip(
                        label: Text(algo.label),
                        selected: _currentAlgorithm == algo,
                        onSelected: (_) => _switchAlgorithm(algo),
                        selectedColor: Color(0xFF0080FF),
                      ))
                  .toList(),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildVisualizationTab() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(8),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text('Live Data Visualization',
                    style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: Color(0xFF00FFFF))),
                Text('$_operationsPerSecond ops/sec',
                    style: const TextStyle(color: Color(0xFF00FF80))),
              ],
            ),
            const SizedBox(height: 16),
            SizedBox(
              height: 300,
              child: CustomPaint(
                painter: DataVisualizationPainter(
                  inputData: _previewLastData,
                  outputData: _previewOutputData,
                  animation: _waveController,
                  colorAnimation: _colorController,
                  dataType: _currentType,
                ),
                size: Size.infinite,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDataTab() {
    return Card(
      child: Padding(
        padding: EdgeInsets.all(8),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Raw Data View',
                style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF00FFFF))),
            SizedBox(height: 16),
            Container(
              height: 300,
              child: Row(
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Input Data',
                            style: TextStyle(
                                fontWeight: FontWeight.bold,
                                color: Color(0xFF0080FF))),
                        SizedBox(height: 8),
                        Expanded(
                          child: Container(
                            padding: EdgeInsets.all(8),
                            decoration: BoxDecoration(
                              border: Border.all(color: Color(0xFF0080FF)),
                              borderRadius: BorderRadius.circular(4),
                            ),
                            child: ListView.builder(
                              itemCount:
                                  _previewInputData.length, // Use safe getter
                              itemBuilder: (context, index) {
                                return Text(
                                  '[$index]: ${_previewInputData[index].toStringAsFixed(3)}',
                                  style: TextStyle(
                                      fontFamily: 'monospace',
                                      color: Color(0xFF00FFFF)),
                                );
                              },
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                  SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Output Data',
                            style: TextStyle(
                                fontWeight: FontWeight.bold,
                                color: Color(0xFF00FF80))),
                        SizedBox(height: 8),
                        Expanded(
                          child: Container(
                            padding: EdgeInsets.all(8),
                            decoration: BoxDecoration(
                              border: Border.all(color: Color(0xFF00FF80)),
                              borderRadius: BorderRadius.circular(4),
                            ),
                            child: ListView.builder(
                              itemCount:
                                  _previewOutputData.length, // Use safe getter
                              itemBuilder: (context, index) {
                                return Text(
                                  '[$index]: ${_previewOutputData[index].toStringAsFixed(3)}',
                                  style: TextStyle(
                                      fontFamily: 'monospace',
                                      color: Color(0xFF00FF80)),
                                );
                              },
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildControls() {
    return Card(
      child: Padding(
        padding: EdgeInsets.all(8),
        child: Column(
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  onPressed: _autoMode ? null : _runKernel,
                  icon: Icon(Icons.play_arrow),
                  label: Text('Execute'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Color(0xFF0080FF),
                    foregroundColor: Colors.white,
                  ),
                ),
                ElevatedButton.icon(
                  onPressed: () {
                    setState(() {
                      _autoMode = !_autoMode;
                    });
                    if (_autoMode) _runKernel();
                  },
                  icon: Icon(_autoMode ? Icons.pause : Icons.autorenew),
                  label: Text(_autoMode ? 'Stop Auto' : 'Auto Mode'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor:
                        _autoMode ? Color(0xFFFF4080) : Color(0xFF00FF80),
                    foregroundColor: Colors.white,
                  ),
                ),
                ElevatedButton.icon(
                  onPressed: _generateInputData,
                  icon: Icon(Icons.shuffle),
                  label: Text('Randomize'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Color(0xFF00FFFF),
                    foregroundColor: Colors.black,
                  ),
                ),
              ],
            ),
            SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: Column(
                    children: [
                      Text('Buffer Size: $_bufferSize',
                          style: TextStyle(color: Color(0xFF00FFFF))),
                      Slider(
                        min: 64,
                        max: 221184,
                        divisions: 108,
                        value: _bufferSize.toDouble(),
                        activeColor: Color(0xFF00FFFF),
                        inactiveColor: Color(0xFF0080FF),
                        onChanged: (value) {
                          setState(() {
                            _bufferSize = value.toInt();
                            _previewLength =
                                _previewLength.clamp(1, _bufferSize);
                          });
                          _createBuffers();
                          _generateInputData();
                        },
                      ),
                    ],
                  ),
                ),
                SizedBox(width: 16),
                Expanded(
                  child: Column(
                    children: [
                      Text('Preview: $_previewLength',
                          style: TextStyle(color: Color(0xFF00FFFF))),
                      Slider(
                        min: 8,
                        max: 4096,
                        divisions: 108,
                        value: _previewLength.toDouble(),
                        activeColor: Color(0xFF00FFFF),
                        inactiveColor: Color(0xFF0080FF),
                        onChanged: (value) {
                          setState(() {
                            _previewLength =
                                value.toInt().clamp(1, _bufferSize);
                            // No more dangerous sublist operations!
                            // The getters handle safe data access automatically
                          });
                        },
                      ),
                    ],
                  ),
                ),
              ],
            ),
            if (_autoMode)
              Column(
                children: [
                  Text(
                      'Animation Speed: ${_animationSpeed.toStringAsFixed(1)}x',
                      style: TextStyle(color: Color(0xFF00FFFF))),
                  Slider(
                    min: 0.1,
                    max: 1000.0,
                    value: _animationSpeed,
                    activeColor: Color(0xFF00FFFF),
                    inactiveColor: Color(0xFF0080FF),
                    onChanged: (value) {
                      setState(() {
                        _animationSpeed = value;
                      });
                    },
                  ),
                ],
              ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Minigpu Type Testing Lab'),
        bottom: TabBar(
          controller: _tabController,
          labelColor: Color(0xFF00FFFF),
          unselectedLabelColor: Color(0xFF0080FF),
          indicatorColor: Color(0xFF00FFFF),
          tabs: [
            Tab(icon: Icon(Icons.timeline), text: 'Chart'),
            Tab(icon: Icon(Icons.data_object), text: 'Raw Data'),
          ],
        ),
      ),
      body: Column(
        children: [
          _buildTypeSelector(),
          _buildAlgorithmSelector(),
          Expanded(
            child: TabBarView(
              controller: _tabController,
              children: [
                SingleChildScrollView(
                  padding: EdgeInsets.all(8),
                  child: _buildVisualizationTab(),
                ),
                SingleChildScrollView(
                  padding: EdgeInsets.all(8),
                  child: _buildDataTab(),
                ),
              ],
            ),
          ),
          _buildControls(),
        ],
      ),
    );
  }
}

class DataVisualizationPainter extends CustomPainter {
  final List<double> inputData;
  final List<double> outputData;
  final Animation<double> animation;
  final Animation<double> colorAnimation;
  final DataTypeDemo dataType;

  DataVisualizationPainter({
    required this.inputData,
    required this.outputData,
    required this.animation,
    required this.colorAnimation,
    required this.dataType,
  }) : super(repaint: animation);

  static Path? _cachedInputPath;
  static Path? _cachedOutputPath;
  static double _cachedMinVal = 0;
  static double _cachedMaxVal = 0;
  static int _cachedDataHash = 0;
  static Size? _cachedGridSize;
  static Picture? _cachedGrid;

  @override
  void paint(Canvas canvas, Size size) {
    if (inputData.isEmpty && outputData.isEmpty) return;
    if (size.width <= 0 || size.height <= 0) return;

    // Check if we can reuse cached paths
    int currentDataHash = inputData.hashCode ^ outputData.hashCode;
    bool needsRecalculation = currentDataHash != _cachedDataHash;

    if (needsRecalculation) {
      _calculatePaths(size);
      _cachedDataHash = currentDataHash;
    }

    // Draw grid ONLY when size changes
    if (_cachedGridSize != size) {
      _cacheGrid(size);
      _cachedGridSize = size;
    }

    // Draw cached grid
    if (_cachedGrid != null) {
      canvas.drawPicture(_cachedGrid!);
    }

    // Use cached paths for drawing
    final paint = Paint()
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;

    // Draw paths
    if (_cachedInputPath != null) {
      paint.color = Color(0xFF0080FF).withOpacity(0.8);
      canvas.drawPath(_cachedInputPath!, paint);
    }

    if (_cachedOutputPath != null) {
      paint.color = Color(0xFF00FFFF);
      canvas.drawPath(_cachedOutputPath!, paint);
    }

    // Draw fewer animated points (every 4th point)
    _drawAnimatedPoints(canvas, size);
  }

  void _cacheGrid(Size size) {
    final recorder = PictureRecorder();
    final gridCanvas = Canvas(recorder);

    final gridPaint = Paint()
      ..color = Color(0xFF0080FF).withOpacity(0.2)
      ..strokeWidth = 1
      ..style = PaintingStyle.stroke;

    const int gridLines = 10;
    final double horizontalSpacing = size.height * 0.8 / gridLines;
    final double verticalSpacing = size.width / gridLines;
    final double yOffset = size.height * 0.1;

    // Draw horizontal lines
    for (int i = 0; i <= gridLines; i++) {
      double y = size.height - (i * horizontalSpacing) - yOffset;
      gridCanvas.drawLine(Offset(0, y), Offset(size.width, y), gridPaint);
    }

    // Draw vertical lines
    for (int i = 0; i <= gridLines; i++) {
      double x = i * verticalSpacing;
      gridCanvas.drawLine(Offset(x, 0), Offset(x, size.height), gridPaint);
    }

    _cachedGrid = recorder.endRecording();
  }

  void _calculatePaths(Size size) {
    // Calculate min/max only when data changes
    List<double> allData = [...inputData, ...outputData];
    if (allData.isEmpty) return;

    _cachedMinVal = allData.reduce((a, b) => a < b ? a : b);
    _cachedMaxVal = allData.reduce((a, b) => a > b ? a : b);

    double range = _cachedMaxVal - _cachedMinVal;
    if (range == 0) range = 1;

    // Build paths once
    _cachedInputPath = _buildPath(inputData, size, _cachedMinVal, range);
    _cachedOutputPath = _buildPath(outputData, size, _cachedMinVal, range);
  }

  Path _buildPath(List<double> data, Size size, double minVal, double range) {
    final path = Path();
    if (data.isEmpty) return path;

    for (int i = 0; i < data.length; i++) {
      final x = (i / (data.length - 1)) * size.width;
      final normalizedValue = (data[i] - minVal) / range;
      final y = size.height -
          (normalizedValue * size.height * 0.8) -
          size.height * 0.1;

      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }
    return path;
  }

  void _drawAnimatedPoints(Canvas canvas, Size size) {
    // Draw fewer points with animation
    for (int i = 0; i < outputData.length; i += 4) {
      // Every 4th point
      final x = (i / (outputData.length - 1)) * size.width;
      final normalizedValue =
          (outputData[i] - _cachedMinVal) / (_cachedMaxVal - _cachedMinVal);
      final y = size.height -
          (normalizedValue * size.height * 0.8) -
          size.height * 0.1;

      final animationOffset = sin(animation.value * 2 * pi + i * 0.3) * 2;

      canvas.drawCircle(
          Offset(x, y),
          3 + animationOffset,
          Paint()
            ..color = Color(0xFF00FF80)
            ..style = PaintingStyle.fill);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}

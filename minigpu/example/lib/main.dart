import 'dart:async';
import 'dart:math';
import 'dart:typed_data';
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
  increment('x + 1'),
  square('x * x'),
  gelu('GELU Activation'),
  sine('sin(x)'),
  matrixAdd('Matrix Addition'),
  convolution('1D Convolution'),
  sorting('Bitonic Sort'),
  reduction('Parallel Sum');

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
  AlgorithmDemo _currentAlgorithm = AlgorithmDemo.increment;
  int _bufferSize = 256;
  int _previewLength = 16;
  bool _autoMode = false;
  bool _isRunning = false;
  double _animationSpeed = 1.0;

  // Visualization data
  List<double> _inputData = [];
  List<double> _outputData = [];
  List<double> _fullInputData = [];
  List<double> _fullOutputData = [];

  // Performance metrics
  int _operationsPerSecond = 0;
  int _totalOperations = 0;
  DateTime _lastOpTime = DateTime.now();

  final Random _random = Random();

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

    // Scale data to reasonable ranges for visualization
    double scaleFactor = _getVisualizationScale();

    switch (_currentAlgorithm) {
      case AlgorithmDemo.sine:
        data = List.generate(
            _bufferSize, (i) => sin(i * 2 * pi / _bufferSize) * scaleFactor);
        break;
      case AlgorithmDemo.matrixAdd:
        data = List.generate(
            _bufferSize, (i) => (i % 16).toDouble() * (scaleFactor / 20));
        break;
      case AlgorithmDemo.convolution:
        data = List.generate(
            _bufferSize, (i) => (_random.nextDouble() - 0.5) * scaleFactor);
        break;
      case AlgorithmDemo.sorting:
        data = List.generate(
            _bufferSize, (i) => _random.nextDouble() * scaleFactor);
        break;
      default:
        data = List.generate(
            _bufferSize, (i) => (_random.nextDouble() - 0.5) * scaleFactor);
    }

    _setBufferData(data);
  }

  double _getVisualizationScale() {
    // Scale data to fit nicely in visualization regardless of type
    switch (_currentType.type) {
      case BufferDataType.int8:
        return 100.0;
      case BufferDataType.uint8:
        return 200.0;
      case BufferDataType.int16:
        return 1000.0;
      case BufferDataType.uint16:
        return 2000.0;
      case BufferDataType.int32:
        return 10000.0;
      case BufferDataType.uint32:
        return 20000.0;
      case BufferDataType.float32:
      case BufferDataType.float64:
        return 500.0;
      default:
        return 500.0;
    }
  }

  void _setBufferData(List<double> data) {
    // Filter valid data
    final validData = data.where((value) => value.isFinite).toList();

    setState(() {
      _fullInputData = List.from(validData);
      _inputData =
          validData.sublist(0, _previewLength.clamp(0, validData.length));
    });

    switch (_currentType.type) {
      case BufferDataType.int8:
        final Int8List typedData = Int8List.fromList(
            validData.map((e) => e.clamp(-128, 127).toInt()).toList());
        _inputBuffer?.setData(typedData, _bufferSize,
            dataType: _currentType.type);
        break;
      case BufferDataType.uint8:
        final Uint8List typedData = Uint8List.fromList(
            validData.map((e) => e.clamp(0, 255).toInt()).toList());
        _inputBuffer?.setData(typedData, _bufferSize,
            dataType: _currentType.type);
        break;
      case BufferDataType.int16:
        final Int16List typedData = Int16List.fromList(
            validData.map((e) => e.clamp(-32768, 32767).toInt()).toList());
        _inputBuffer?.setData(typedData, _bufferSize,
            dataType: _currentType.type);
        break;
      case BufferDataType.uint16:
        final Uint16List typedData = Uint16List.fromList(
            validData.map((e) => e.clamp(0, 65535).toInt()).toList());
        _inputBuffer?.setData(typedData, _bufferSize,
            dataType: _currentType.type);
        break;
      case BufferDataType.int32:
        final Int32List typedData = Int32List.fromList(validData
            .map((e) => e.clamp(-2147483648, 2147483647).toInt())
            .toList());
        _inputBuffer?.setData(typedData, _bufferSize,
            dataType: _currentType.type);
        break;
      case BufferDataType.uint32:
        final Uint32List typedData = Uint32List.fromList(
            validData.map((e) => e.clamp(0, 4294967295).toInt()).toList());
        _inputBuffer?.setData(typedData, _bufferSize,
            dataType: _currentType.type);
        break;
      case BufferDataType.float32:
        final Float32List typedData = Float32List.fromList(validData);
        _inputBuffer?.setData(typedData, _bufferSize,
            dataType: _currentType.type);
        break;
      case BufferDataType.float64:
        final Float64List typedData = Float64List.fromList(validData);
        _inputBuffer?.setData(typedData, _bufferSize,
            dataType: _currentType.type);
        break;
      default:
        final Float32List typedData = Float32List.fromList(validData);
        _inputBuffer?.setData(typedData, _bufferSize,
            dataType: BufferDataType.float32);
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

    switch (_currentAlgorithm) {
      case AlgorithmDemo.increment:
        return 'out[idx] = x + ${isFloat ? '1.0' : '1'};';
      case AlgorithmDemo.square:
        return 'out[idx] = x * x;';
      case AlgorithmDemo.gelu:
        if (isFloat) {
          return '''
        let gelu_factor = 0.7978845608028654; // sqrt(2.0 / PI)
        let cubic = x + 0.044715 * x * x * x;
        let tanh_arg = gelu_factor * cubic;
        out[idx] = 0.5 * x * (1.0 + tanh(tanh_arg));''';
        } else {
          return 'out[idx] = select(x, x + 1, x > 0);'; // Simplified for integers
        }
      case AlgorithmDemo.sine:
        if (isFloat) {
          return 'out[idx] = sin(x * 0.1) * 100.0;';
        } else {
          return 'out[idx] = x + (x % 10);'; // Simplified pattern for integers
        }
      case AlgorithmDemo.matrixAdd:
        return 'out[idx] = x + inp[(idx + 1) % arrayLength(&inp)];';
      case AlgorithmDemo.convolution:
        return '''
        var sum = x;
        if (idx > 0) { sum += inp[idx - 1]; }
        if (idx < arrayLength(&inp) - 1) { sum += inp[idx + 1]; }
        out[idx] = sum / ${isFloat ? '3.0' : '3'};''';
      case AlgorithmDemo.sorting:
        return 'out[idx] = x; // Copy for visualization';
      case AlgorithmDemo.reduction:
        return 'out[idx] = x + ${isFloat ? '1.0' : '1'}; // Step 1 of reduction';
    }
  }

  Future<void> _runKernel() async {
    if (!_minigpu.isInitialized ||
        _inputBuffer == null ||
        _outputBuffer == null) return;

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
            _fullOutputData = outputData.map((e) => e.toDouble()).toList();
            _outputData = _fullOutputData
                .take(_previewLength)
                .where((value) => value.isFinite)
                .toList();
          });
          break;
        case BufferDataType.uint8:
          final Uint8List outputData = Uint8List(_bufferSize);
          await _outputBuffer?.read(outputData, _bufferSize,
              dataType: _currentType.type);
          setState(() {
            _fullOutputData = outputData.map((e) => e.toDouble()).toList();
            _outputData = _fullOutputData
                .take(_previewLength)
                .where((value) => value.isFinite)
                .toList();
          });
          break;
        case BufferDataType.int16:
          final Int16List outputData = Int16List(_bufferSize);
          await _outputBuffer?.read(outputData, _bufferSize,
              dataType: _currentType.type);
          setState(() {
            _fullOutputData = outputData.map((e) => e.toDouble()).toList();
            _outputData = _fullOutputData
                .take(_previewLength)
                .where((value) => value.isFinite)
                .toList();
          });
          break;
        case BufferDataType.uint16:
          final Uint16List outputData = Uint16List(_bufferSize);
          await _outputBuffer?.read(outputData, _bufferSize,
              dataType: _currentType.type);
          setState(() {
            _fullOutputData = outputData.map((e) => e.toDouble()).toList();
            _outputData = _fullOutputData
                .take(_previewLength)
                .where((value) => value.isFinite)
                .toList();
          });
          break;
        case BufferDataType.float32:
          final Float32List outputData = Float32List(_bufferSize);
          await _outputBuffer?.read(outputData, _bufferSize,
              dataType: _currentType.type);
          setState(() {
            _fullOutputData = outputData.map((e) => e.toDouble()).toList();
            _outputData = _fullOutputData
                .take(_previewLength)
                .where((value) => value.isFinite)
                .toList();
          });
          break;
        default:
          final Float32List outputData = Float32List(_bufferSize);
          await _outputBuffer?.read(outputData, _bufferSize,
              dataType: BufferDataType.float32);
          setState(() {
            _fullOutputData = outputData.map((e) => e.toDouble()).toList();
            _outputData = _fullOutputData
                .take(_previewLength)
                .where((value) => value.isFinite)
                .toList();
          });
      }

      _updatePerformanceMetrics();

      if (_autoMode && mounted) {
        Future.delayed(Duration(milliseconds: (1000 / _animationSpeed).round()),
            () {
          if (mounted && _autoMode) {
            _generateInputData();
            _runKernel();
          }
        });
      }
    } catch (e) {
      print('Shader execution error: $e');
      setState(() {
        _inputData = [];
        _outputData = [];
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
    _createBuffers();
    _generateInputData();
  }

  void _switchAlgorithm(AlgorithmDemo newAlgorithm) {
    setState(() {
      _currentAlgorithm = newAlgorithm;
    });
    _generateInputData();
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
        padding: EdgeInsets.all(8),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text('Live Data Visualization',
                    style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: Color(0xFF00FFFF))),
                Text('${_operationsPerSecond} ops/sec',
                    style: TextStyle(color: Color(0xFF00FF80))),
              ],
            ),
            SizedBox(height: 16),
            Container(
              height: 300,
              child: CustomPaint(
                painter: DataVisualizationPainter(
                  inputData: _inputData,
                  outputData: _outputData,
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
                              itemCount: _inputData.length,
                              itemBuilder: (context, index) {
                                return Text(
                                  '[$index]: ${_inputData[index].toStringAsFixed(3)}',
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
                              itemCount: _outputData.length,
                              itemBuilder: (context, index) {
                                return Text(
                                  '[$index]: ${_outputData[index].toStringAsFixed(3)}',
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
                  onPressed: _runKernel,
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
                        max: 2048,
                        divisions: 31,
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
                        max: 64,
                        divisions: 7,
                        value: _previewLength.toDouble(),
                        activeColor: Color(0xFF00FFFF),
                        inactiveColor: Color(0xFF0080FF),
                        onChanged: (value) {
                          setState(() {
                            _previewLength =
                                value.toInt().clamp(1, _bufferSize);
                            _inputData = _fullInputData.sublist(0,
                                _previewLength.clamp(0, _fullInputData.length));
                            _outputData = _fullOutputData.sublist(
                                0,
                                _previewLength.clamp(
                                    0, _fullOutputData.length));
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

  @override
  void paint(Canvas canvas, Size size) {
    if (inputData.isEmpty && outputData.isEmpty) return;
    if (size.width <= 0 || size.height <= 0) return;

    final paint = Paint()
      ..strokeWidth = 3
      ..style = PaintingStyle.stroke;

    final width = size.width;
    final height = size.height;
    final centerY = height / 2;

    // Calculate data range for proper scaling
    List<double> allData = [...inputData, ...outputData];
    if (allData.isEmpty) return;

    double minVal = allData.reduce((a, b) => a < b ? a : b);
    double maxVal = allData.reduce((a, b) => a > b ? a : b);

    // Add padding to range
    double range = maxVal - minVal;
    if (range == 0) range = 1;
    minVal -= range * 0.1;
    maxVal += range * 0.1;
    range = maxVal - minVal;

    // Draw input data (electric blue)
    if (inputData.isNotEmpty && inputData.length > 1) {
      paint.color = Color(0xFF0080FF).withOpacity(0.8);
      final inputPath = Path();

      for (int i = 0; i < inputData.length; i++) {
        final x = (i / (inputData.length - 1)) * width;
        final value = inputData[i];
        if (!value.isFinite) continue;

        // Scale to fit chart area
        final normalizedValue = (value - minVal) / range;
        final y = height - (normalizedValue * height * 0.8) - height * 0.1;

        if (!x.isFinite || !y.isFinite) continue;

        if (i == 0) {
          inputPath.moveTo(x, y);
        } else {
          inputPath.lineTo(x, y);
        }

        // Draw data points with electric glow
        final animationOffset = animation.value.isFinite
            ? sin(animation.value * 2 * pi + i * 0.5) * 2
            : 0.0;

        // Glow effect
        canvas.drawCircle(
            Offset(x, y),
            6 + animationOffset,
            Paint()
              ..color = Color(0xFF0080FF).withOpacity(0.3)
              ..style = PaintingStyle.fill);

        canvas.drawCircle(
            Offset(x, y),
            3 + animationOffset * 0.5,
            Paint()
              ..color = Color(0xFF00FFFF)
              ..style = PaintingStyle.fill);
      }
      canvas.drawPath(inputPath, paint);
    }

    // Draw output data (electric cyan)
    if (outputData.isNotEmpty && outputData.length > 1) {
      paint.color = Color(0xFF00FFFF);
      final outputPath = Path();

      for (int i = 0; i < outputData.length; i++) {
        final x = (i / (outputData.length - 1)) * width;
        final value = outputData[i];
        if (!value.isFinite) continue;

        // Scale to fit chart area
        final normalizedValue = (value - minVal) / range;
        final y = height - (normalizedValue * height * 0.8) - height * 0.1;

        if (!x.isFinite || !y.isFinite) continue;

        if (i == 0) {
          outputPath.moveTo(x, y);
        } else {
          outputPath.lineTo(x, y);
        }

        // Animated output points with electric effect
        final animationOffset = animation.value.isFinite
            ? sin(animation.value * 2 * pi + i * 0.3) * 3
            : 0.0;

        // Glow effect
        canvas.drawCircle(
            Offset(x, y),
            8 + animationOffset,
            Paint()
              ..color = Color(0xFF00FFFF).withOpacity(0.2)
              ..style = PaintingStyle.fill);

        canvas.drawCircle(
            Offset(x, y),
            4 + animationOffset * 0.5,
            Paint()
              ..color = Color(0xFF00FF80)
              ..style = PaintingStyle.fill);
      }
      canvas.drawPath(outputPath, paint);
    }

    // Draw electric grid
    final gridPaint = Paint()
      ..color = Color(0xFF004080).withOpacity(0.5)
      ..strokeWidth = 1;

    // Horizontal lines
    for (int i = 0; i <= 4; i++) {
      final y = (i / 4) * height;
      if (y.isFinite) {
        canvas.drawLine(Offset(0, y), Offset(width, y), gridPaint);
      }
    }

    // Vertical lines
    for (int i = 0; i <= 8; i++) {
      final x = (i / 8) * width;
      if (x.isFinite) {
        canvas.drawLine(Offset(x, 0), Offset(x, height), gridPaint);
      }
    }

    // Draw scale labels
    final textPainter = TextPainter(
      textAlign: TextAlign.center,
      textDirection: TextDirection.ltr,
    );

    // Min value label
    textPainter.text = TextSpan(
      text: minVal.toStringAsFixed(1),
      style: TextStyle(color: Color(0xFF00FFFF), fontSize: 12),
    );
    textPainter.layout();
    textPainter.paint(canvas, Offset(5, height - 20));

    // Max value label
    textPainter.text = TextSpan(
      text: maxVal.toStringAsFixed(1),
      style: TextStyle(color: Color(0xFF00FFFF), fontSize: 12),
    );
    textPainter.layout();
    textPainter.paint(canvas, Offset(5, 5));
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}

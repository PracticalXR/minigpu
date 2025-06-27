import 'dart:typed_data';
import 'package:minigpu/minigpu.dart';
import 'gpu_helpers.dart';

int _elementSize(BufferDataType type) {
  switch (type) {
    case BufferDataType.int8:
    case BufferDataType.uint8:
      return 1;
    case BufferDataType.float16:
    case BufferDataType.int16:
    case BufferDataType.uint16:
      return 2;
    case BufferDataType.int32:
    case BufferDataType.uint32:
    case BufferDataType.float32:
      return 4;
    case BufferDataType.int64:
    case BufferDataType.uint64:
    case BufferDataType.float64:
      return 8;
  }
}

/// A helper that creates (or reuses) a default GPU context.
class DefaultMinigpu {
  static final instance = Minigpu();
}

final Finalizer<Buffer> _bufferFinalizer = Finalizer(
  (buffer) => buffer.destroy(),
);

/// A generic tensor that works with a specific [TypedData] type.
class Tensor<T extends TypedData> {
  /// The shape in terms of dimensions.
  final List<int> shape;

  /// Total number of elements.
  final int size;
  int get rank => shape.length;
  int get elementSize => _elementSize(dataType);
  int get byteSize => size * elementSize;
  int get elementCount => shape.reduce((a, b) => a * b);
  final Minigpu gpu;
  late Buffer buffer;

  /// The underlying data type.
  final BufferDataType dataType;

  ComputeShader? activeShader;

  // Private constructor.
  Tensor._(
    this.shape, {
    required this.gpu,
    T? data,
    this.dataType = BufferDataType.float32,
  }) : size = shape.reduce((a, b) => a * b) {
    final int byteSize = size * _elementSize(dataType);
    buffer = gpu.createBuffer(byteSize, dataType);
    _bufferFinalizer.attach(this, buffer);
    if (data != null) {
      if (data.lengthInBytes ~/ _elementSize(dataType) != size) {
        throw Exception(
          "Provided data length (${data.lengthInBytes ~/ _elementSize(dataType)}) does not match tensor size ($size)",
        );
      }
      buffer.write(data, size, dataType: dataType);
    } else {
      // Initialize with zeros.
      if (T == TypedData) {
        buffer.write(Float32List(size), size, dataType: dataType);
      } else if (T == Int8List) {
        buffer.write(Int8List(size), size, dataType: dataType);
      } else if (T == Int16List) {
        buffer.write(Int16List(size), size, dataType: dataType);
      } else if (T == Int32List) {
        buffer.write(Int32List(size), size, dataType: dataType);
      } else if (T == Int64List) {
        buffer.write(Int64List(size), size, dataType: dataType);
      } else if (T == Uint8List) {
        buffer.write(Uint8List(size), size, dataType: dataType);
      } else if (T == Uint16List) {
        buffer.write(Uint16List(size), size, dataType: dataType);
      } else if (T == Uint32List) {
        buffer.write(Uint32List(size), size, dataType: dataType);
      } else if (T == Float32List) {
        buffer.write(Float32List(size), size, dataType: dataType);
      } else if (T == Float64List) {
        buffer.write(Float64List(size), size, dataType: dataType);
      } else {
        throw Exception("Unsupported TypedData type: ${T.toString()}");
      }
    }
  }

  /// Asynchronous factory to create a tensor.
  static Future<Tensor<T>> create<T extends TypedData>(
    List<int> shape, {
    Minigpu? gpu,
    T? data,
    BufferDataType dataType = BufferDataType.float32,
  }) async {
    gpu = gpu ?? DefaultMinigpu.instance;
    if (!gpu.isInitialized) {
      print("Initializing GPU context...");
      await gpu.init();
    }
    return Tensor._(shape, gpu: gpu, data: data, dataType: dataType);
  }

  /// Returns the data from the GPU buffer as type T.
  Future<T> getData() async {
    late T result;
    if (T == TypedData) {
      result = Float32List(size) as T;
    } else if (T == Int8List) {
      result = Int8List(size) as T;
    } else if (T == Int16List) {
      result = Int16List(size) as T;
    } else if (T == Int32List) {
      result = Int32List(size) as T;
    } else if (T == Int64List) {
      result = Int64List(size) as T;
    } else if (T == Uint8List) {
      result = Uint8List(size) as T;
    } else if (T == Uint16List) {
      result = Uint16List(size) as T;
    } else if (T == Uint32List) {
      result = Uint32List(size) as T;
    } else if (T == Float32List) {
      result = Float32List(size) as T;
    } else if (T == Float64List) {
      result = Float64List(size) as T;
    } else {
      throw Exception("Unsupported TypedData type: ${T.toString()}");
    }
    await buffer.read(result, size, dataType: dataType);
    return result;
  }

  /// Writes [data] of type T to the GPU buffer.
  Future<void> write(T data) async {
    await buffer.write(data, size, dataType: dataType);
  }

  /// Destroys the tensor's GPU buffer.
  void destroy() {
    _bufferFinalizer.detach(this);
    buffer.destroy();
  }

  /// Creates a tensor from an existing buffer.
  Tensor.fromBuffer(
    this.buffer,
    this.shape, {
    Minigpu? gpu,
    this.dataType = BufferDataType.float32,
  }) : gpu = gpu ?? DefaultMinigpu.instance,
       size = shape.reduce((a, b) => a * b) {
    _bufferFinalizer.attach(this, buffer);
  }

  /// Creates a tensor filled with zeros using GPU shader.
  static Future<Tensor<T>> zeros<T extends TypedData>(
    List<int> shape, {
    Minigpu? gpu,
    BufferDataType dataType = BufferDataType.float32,
  }) async {
    gpu = gpu ?? DefaultMinigpu.instance;
    if (!gpu.isInitialized) {
      print("Initializing GPU context...");
      await gpu.init();
    }

    // Use private constructor with explicit cast
    final tensor = Tensor._(shape, gpu: gpu, dataType: dataType) as Tensor<T>;
    final size = tensor.size;

    final shaderTemplate = '''
@group(0) @binding(0) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < \${size}u) {
    output[i] = \${zeroValue};
  }
}
''';

    final ComputeShader shader = gpu.createComputeShader();
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'size': size,
      'zeroValue': getZeroValue(dataType),
    });
    shader.loadKernelString(shaderCode);
    shader.setBuffer('output', tensor.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();

    return tensor;
  }

  /// Creates a tensor filled with random values using GPU shader with CPU-generated seed.
  static Future<Tensor<T>> random<T extends TypedData>(
    List<int> shape, {
    Minigpu? gpu,
    BufferDataType dataType = BufferDataType.float32,
    double min = 0.0,
    double max = 1.0,
    int? seed,
  }) async {
    gpu = gpu ?? DefaultMinigpu.instance;
    if (!gpu.isInitialized) {
      print("Initializing GPU context...");
      await gpu.init();
    }

    final tensor = Tensor._(shape, gpu: gpu, dataType: dataType) as Tensor<T>;
    final size = tensor.size;
    final actualSeed = seed ?? DateTime.now().millisecondsSinceEpoch;

    final shaderTemplate = '''
@group(0) @binding(0) var<storage, read_write> output: array<f32>;

// Simple LCG random number generator
fn rand(state: ptr<function, u32>) -> f32 {
  *state = (*state * 1664525u + 1013904223u);
  return f32(*state) / 4294967296.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < \${size}u) {
    var rng_state: u32 = \${seed}u + i;
    let random_val = rand(&rng_state);
    let scaled_val = \${min} + random_val * \${range};
    output[i] = \${castValue};
  }
}
''';

    final ComputeShader shader = gpu.createComputeShader();
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'size': size,
      'seed': actualSeed,
      'min': min,
      'range': max - min,
      'castValue': getCastExpression('scaled_val', dataType),
    });
    shader.loadKernelString(shaderCode);
    shader.setBuffer('output', tensor.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();

    return tensor;
  }

  /// Creates a tensor from byte data using GPU shader for type conversion.
  /// The byte data format should be: [rank][dim1][dim2]...[dimN][data...]
  /// where rank is 4 bytes (int32), each dimension is 4 bytes (int32),
  /// followed by the actual tensor data.
  static Future<Tensor<T>> fromBytes<T extends TypedData>(
    Uint8List bytes, {
    Minigpu? gpu,
    BufferDataType dataType = BufferDataType.float32,
  }) async {
    gpu = gpu ?? DefaultMinigpu.instance;
    if (!gpu.isInitialized) {
      print("Initializing GPU context...");
      await gpu.init();
    }

    // Parse header: rank (4 bytes) + dimensions (4 bytes each)
    if (bytes.length < 4) {
      throw Exception("Byte data too short to contain rank information");
    }

    final byteData = ByteData.sublistView(bytes);
    final rank = byteData.getInt32(0, Endian.little);

    if (rank <= 0 || rank > 8) {
      throw Exception("Invalid tensor rank: $rank");
    }

    final headerSize = 4 + (rank * 4); // rank + dimensions
    if (bytes.length < headerSize) {
      throw Exception("Byte data too short to contain shape information");
    }

    // Parse shape
    final shape = <int>[];
    for (int i = 0; i < rank; i++) {
      final dim = byteData.getInt32(4 + (i * 4), Endian.little);
      if (dim <= 0) {
        throw Exception("Invalid dimension at index $i: $dim");
      }
      shape.add(dim);
    }

    // Calculate expected data size
    final size = shape.reduce((a, b) => a * b);
    final elementSize = _elementSize(dataType);
    final expectedDataBytes = size * elementSize;
    final actualDataBytes = bytes.length - headerSize;

    if (actualDataBytes != expectedDataBytes) {
      throw Exception(
        "Data size mismatch: expected $expectedDataBytes bytes, got $actualDataBytes bytes",
      );
    }

    // Create tensor
    final tensor =
        Tensor<T>._(shape, gpu: gpu, dataType: dataType) as Tensor<T>;

    // Extract data portion
    final dataBytes = bytes.sublist(headerSize);

    // Create temporary buffer for byte data
    final byteBuffer = gpu.createBuffer(dataBytes.length, BufferDataType.uint8);
    await byteBuffer.write(
      dataBytes,
      dataBytes.length,
      dataType: BufferDataType.uint8,
    );

    final shaderTemplate = '''
@group(0) @binding(0) var<storage, read_write> input_bytes: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i: u32 = gid.x;
  if (i < \${size}u) {
    \${conversionCode}
  }
}
''';

    final ComputeShader shader = gpu.createComputeShader();
    final shaderCode = prepareShader(shaderTemplate, dataType, {
      'size': size,
      'conversionCode': getByteConversionCode(dataType),
    });
    shader.loadKernelString(shaderCode);
    shader.setBuffer('input_bytes', byteBuffer);
    shader.setBuffer('output', tensor.buffer);
    int workgroups = (size + 255) ~/ 256;
    await shader.dispatch(workgroups, 1, 1);
    shader.destroy();
    byteBuffer.destroy();

    return tensor;
  }

  /// Saves tensor to bytes with shape information.
  /// Format: [rank][dim1][dim2]...[dimN][data...]
  Future<Uint8List> toBytes() async {
    final data = await getData();
    final headerSize = 4 + (rank * 4);
    final dataSize = data.lengthInBytes;
    final totalSize = headerSize + dataSize;

    final result = Uint8List(totalSize);
    final byteData = ByteData.sublistView(result);

    // Write rank
    byteData.setInt32(0, rank, Endian.little);

    // Write dimensions
    for (int i = 0; i < rank; i++) {
      byteData.setInt32(4 + (i * 4), shape[i], Endian.little);
    }

    // Write data
    final dataBytes = data.buffer.asUint8List(
      data.offsetInBytes,
      data.lengthInBytes,
    );
    result.setRange(headerSize, totalSize, dataBytes);

    return result;
  }
}

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

  /// For buffer views (see [Tensor.view]): strong reference to the tensor
  /// that owns [buffer], so the owner cannot be garbage-collected (and its
  /// finalizer cannot destroy the shared buffer) while this view is
  /// reachable.
  Tensor? _viewParent;

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
    }
    // No data: WebGPU zero-initializes buffer contents, so the previous
    // CPU-side zero upload was a redundant full-tensor PCIe transfer paid by
    // EVERY op-result allocation.  Skip it.
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
    final effectiveType = _inferDataTypeFor<T>(dataType);
    return Tensor<T>._(shape, gpu: gpu, data: data, dataType: effectiveType);
  }

  /// Returns the data from the GPU buffer as type T.
  ///
  /// Pass [into] to reuse an already-allocated typed list instead of
  /// allocating a fresh one on every call. This is the hot path for streaming
  /// pipelines (one readback per frame): supplying a persistent scratch list
  /// eliminates per-frame heap allocation and the associated GC pressure.
  /// [into] must be the correct list type for `T` and have `length >= size`.
  Future<T> getData({T? into}) async {
    late T result;
    if (into != null) {
      if ((into as TypedData).lengthInBytes <
          size * _elementSizeBytes(dataType)) {
        throw ArgumentError(
          'getData(into:) target is too small: needs $size elements',
        );
      }
      result = into;
    } else if (T == TypedData) {
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

  static int _elementSizeBytes(BufferDataType t) {
    switch (t) {
      case BufferDataType.int8:
      case BufferDataType.uint8:
        return 1;
      case BufferDataType.int16:
      case BufferDataType.uint16:
      case BufferDataType.float16:
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

  /// Writes [data] of type T to the GPU buffer.
  Future<void> write(T data) async {
    await buffer.write(data, size, dataType: dataType);
  }

  /// Destroys the tensor's GPU buffer.
  ///
  /// For views (see [Tensor.view]) this releases the parent reference but
  /// does NOT destroy the shared buffer — the owning tensor does that.
  void destroy() {
    activeShader?.destroy();
    activeShader = null;
    if (_viewParent != null) {
      _viewParent = null;
      return;
    }
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

  /// Creates a non-owning tensor view over an existing [buffer].
  ///
  /// Unlike [Tensor.fromBuffer], this constructor does **not** attach a
  /// finalizer.  The caller is responsible for keeping [buffer] alive and
  /// destroying it when appropriate.  The Tensor itself must **not** be used
  /// after the underlying buffer has been destroyed.
  Tensor.external(
    this.buffer,
    this.shape, {
    Minigpu? gpu,
    this.dataType = BufferDataType.float32,
  }) : gpu = gpu ?? DefaultMinigpu.instance,
       size = shape.reduce((a, b) => a * b);

  /// Creates a non-owning view over [parent]'s buffer with a new [shape].
  ///
  /// The view attaches NO finalizer (attaching a second finalizer to a shared
  /// buffer — what `Tensor.fromBuffer` does — leads to double-destroy).
  /// Instead it retains [parent], so the buffer cannot be finalized while the
  /// view is reachable.  Explicitly calling `parent.destroy()` still
  /// invalidates the view — the caller owns that ordering.
  Tensor.view(Tensor parent, this.shape)
    : buffer = parent.buffer,
      gpu = parent.gpu,
      dataType = parent.dataType,
      size = shape.reduce((a, b) => a * b),
      _viewParent = parent {
    if (size != parent.size) {
      throw Exception(
        "View shape $shape (size $size) does not match parent size ${parent.size}",
      );
    }
  }

  /// Whether this tensor is a non-owning view over another tensor's buffer.
  bool get isView => _viewParent != null;

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

    final effectiveType = _inferDataTypeFor<T>(dataType);

    // WebGPU zero-initializes buffer contents — a fresh tensor IS zeros.
    return Tensor<T>._(shape, gpu: gpu, dataType: effectiveType);
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

    final effectiveType = _inferDataTypeFor<T>(dataType);

    final tensor = Tensor<T>._(shape, gpu: gpu, dataType: effectiveType);
    final size = tensor.size;
    final actualSeed = seed ?? DateTime.now().millisecondsSinceEpoch;

    final shaderTemplate = '''
@group(0) @binding(0) var<storage, read_write> output: array<f32>;

// PCG-style stateless hash: adjacent (seed, i) pairs decorrelate fully,
// unlike the previous single-step LCG on seed+i.
fn pcg_hash(input: u32) -> u32 {
  var state: u32 = input * 747796405u + 2891336453u;
  let word: u32 = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i < \${size}u) {
    let random_val = f32(pcg_hash(\${seed}u ^ (i * 0x9E3779B9u))) / 4294967296.0;
    let scaled_val = \${min} + random_val * \${range};
    output[i] = \${castValue};
  }
}
''';

    // Seed is baked into the source, so a per-call shader (not the cache) is
    // correct here — caching would fill the cache with dead seed variants.
    final ComputeShader shader = gpu.createComputeShader();
    final shaderCode = prepareShader(shaderTemplate, effectiveType, {
      'size': size,
      'seed': actualSeed & 0xFFFFFFFF,
      'min': min,
      'range': max - min,
      'castValue': getCastExpression('scaled_val', effectiveType),
    });
    try {
      shader.loadKernelString(shaderCode);
      shader.setBuffer('output', tensor.buffer);
      await shader.dispatchLinear(size);
    } finally {
      shader.destroy();
    }

    return tensor;
  }

  /// Creates a tensor from byte data using GPU shader for type conversion.
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
    final effectiveType = _inferDataTypeFor<T>(dataType);
    final tensor = Tensor<T>._(shape, gpu: gpu, dataType: effectiveType);

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
fn main(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let i: u32 = gid.x + gid.y * (nwg.x * 256u);
  if (i < \${size}u) {
    \${conversionCode}
  }
}
''';

    final shaderCode = prepareShader(shaderTemplate, effectiveType, {
      'size': size,
      'conversionCode': getByteConversionCode(effectiveType),
    });
    final shader = gpu.cachedShader(shaderCode);
    shader.setBuffer('input_bytes', byteBuffer);
    shader.setBuffer('output', tensor.buffer);
    await shader.dispatchLinear(size);
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

  static BufferDataType _inferDataTypeFor<T extends TypedData>(
    BufferDataType requested,
  ) {
    // Honor explicit non-default request
    if (requested != BufferDataType.float32) return requested;

    // Infer from T when caller didn't override (default was float32)
    if (T == Int8List) return BufferDataType.int8;
    if (T == Uint8List) return BufferDataType.uint8;
    if (T == Int16List) return BufferDataType.int16;
    if (T == Uint16List) return BufferDataType.uint16;
    if (T == Int32List) return BufferDataType.int32;
    if (T == Uint32List) return BufferDataType.uint32;
    if (T == Float32List) return BufferDataType.float32;
    if (T == Float64List) return BufferDataType.float64;

    // Fallback
    return requested;
  }
}

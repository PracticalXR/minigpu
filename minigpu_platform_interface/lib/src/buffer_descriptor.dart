/// Explicit byte allocation metadata for ordinary GPU buffers.
library;

abstract final class BufferUsage {
  static const int copySrc = 4;
  static const int copyDst = 8;
  static const int index = 16;
  static const int vertex = 32;
  static const int uniform = 64;
  static const int storage = 128;
  static const int legacy = copySrc | copyDst | storage;
  static const int known =
      copySrc | copyDst | index | vertex | uniform | storage;
}

final class BufferDescriptor {
  BufferDescriptor({required this.sizeBytes, required this.usage}) {
    validate();
  }
  final int sizeBytes;
  final int usage;
  void validate() {
    if (sizeBytes < 4 ||
        sizeBytes > 2147483644 ||
        sizeBytes % 4 != 0 ||
        usage == 0 ||
        usage & ~BufferUsage.known != 0) {
      throw ArgumentError('Invalid buffer descriptor');
    }
  }
}

enum BufferBindingKind {
  storage(0),
  readOnlyStorage(1),
  uniform(2);

  const BufferBindingKind(this.wireCode);

  final int wireCode;
}

/// Explicit byte-range metadata for a compute buffer binding.
final class BufferBindingDescriptor {
  BufferBindingDescriptor({
    required this.kind,
    required this.sizeBytes,
    this.offsetBytes = 0,
  }) {
    validate();
  }

  final BufferBindingKind kind;
  final int sizeBytes;
  final int offsetBytes;

  void validate() {
    if (sizeBytes < 4 ||
        sizeBytes > 2147483644 ||
        sizeBytes % 4 != 0 ||
        offsetBytes < 0 ||
        offsetBytes > 2147483644 ||
        offsetBytes % 4 != 0) {
      throw ArgumentError('Invalid buffer binding descriptor');
    }
  }
}

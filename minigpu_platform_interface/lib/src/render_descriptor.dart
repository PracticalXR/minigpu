library;

import 'dart:collection';
import 'dart:convert';

import 'texture_descriptor.dart';

enum RenderBlendMode {
  opaque(0),
  alpha(1),
  premultipliedAlpha(2),
  additive(3);

  const RenderBlendMode(this.wireCode);
  final int wireCode;
}

enum PrimitiveTopology {
  triangleList(0),
  lineList(1),
  pointList(2);

  const PrimitiveTopology(this.wireCode);
  final int wireCode;
}

enum FrontFace {
  ccw(0),
  cw(1);

  const FrontFace(this.wireCode);
  final int wireCode;
}

enum CullMode {
  none(0),
  front(1),
  back(2);

  const CullMode(this.wireCode);
  final int wireCode;
}

enum VertexStepMode {
  vertex(0),
  instance(1);

  const VertexStepMode(this.wireCode);
  final int wireCode;
}

enum VertexFormat {
  float32(0, 4),
  float32x2(1, 8),
  float32x3(2, 12),
  float32x4(3, 16),
  uint32(4, 4),
  uint32x2(5, 8),
  uint32x4(6, 16),
  sint32(7, 4),
  unorm8x4(8, 4);

  const VertexFormat(this.wireCode, this.bytes);
  final int wireCode, bytes;
}

enum IndexFormat {
  uint16(0, 2),
  uint32(1, 4);

  const IndexFormat(this.wireCode, this.bytes);
  final int wireCode, bytes;
}

enum RenderBindingKind {
  uniformBuffer(0),
  readOnlyStorageBuffer(1),
  sampledFloat(2),
  sampledUnfilterableFloat(3),
  sampledDepth(4),
  filteringSampler(5),
  nonFilteringSampler(6),
  comparisonSampler(7);

  const RenderBindingKind(this.wireCode);
  final int wireCode;
}

final class VertexAttribute {
  VertexAttribute({
    required this.shaderLocation,
    required this.offset,
    required this.format,
  }) {
    validate();
  }
  final int shaderLocation, offset;
  final VertexFormat format;
  void validate() {
    final alignment = format.bytes < 4 ? format.bytes : 4;
    if (shaderLocation < 0 ||
        shaderLocation > 31 ||
        offset < 0 ||
        offset > 2048 ||
        offset % alignment != 0)
      throw ArgumentError('Invalid vertex attribute');
  }
}

final class VertexBufferLayout {
  VertexBufferLayout({
    required this.arrayStride,
    this.stepMode = VertexStepMode.vertex,
    required List<VertexAttribute> attributes,
  }) : attributes = UnmodifiableListView(List.of(attributes)) {
    validate();
  }
  final int arrayStride;
  final VertexStepMode stepMode;
  final List<VertexAttribute> attributes;
  void validate() {
    if (arrayStride < 4 ||
        arrayStride > 2048 ||
        arrayStride % 4 != 0 ||
        attributes.length > 16)
      throw ArgumentError('Invalid vertex layout');
    for (final a in attributes) {
      a.validate();
      if (a.offset > arrayStride || a.format.bytes > arrayStride - a.offset)
        throw ArgumentError('Vertex attribute exceeds stride');
    }
  }
}

final class RenderBindingLayout {
  RenderBindingLayout({
    required this.slot,
    required this.kind,
    required this.visibility,
    this.viewDimension = TextureViewDimension.d2,
    this.minBindingSize = 0,
  }) {
    validate();
  }
  final int slot, visibility, minBindingSize;
  final RenderBindingKind kind;
  final TextureViewDimension viewDimension;
  bool get _buffer =>
      kind == RenderBindingKind.uniformBuffer ||
      kind == RenderBindingKind.readOnlyStorageBuffer;
  bool get _texture =>
      kind == RenderBindingKind.sampledFloat ||
      kind == RenderBindingKind.sampledUnfilterableFloat ||
      kind == RenderBindingKind.sampledDepth;
  void validate() {
    if (slot < 0 || slot > 31 || visibility <= 0 || visibility & ~3 != 0)
      throw ArgumentError('Invalid render binding');
    if (_buffer) {
      if (minBindingSize <= 0 ||
          minBindingSize % 4 != 0 ||
          minBindingSize > 2147483644)
        throw ArgumentError('Invalid buffer binding size');
    } else if (minBindingSize != 0) {
      throw ArgumentError('Only buffer bindings have a minimum size');
    }
    if (!_texture && viewDimension != TextureViewDimension.d2)
      throw ArgumentError('Only texture bindings have a view dimension');
  }
}

final class RenderPipelineDescriptor {
  RenderPipelineDescriptor({
    required this.code,
    this.vertexEntry = 'vsMain',
    this.fragmentEntry = 'fsMain',
    required this.colorFormat,
    this.blend = RenderBlendMode.opaque,
    this.topology = PrimitiveTopology.triangleList,
    this.frontFace = FrontFace.ccw,
    this.cullMode = CullMode.none,
    this.depthEnabled = false,
    this.depthWrite = false,
    this.depthCompare = SamplerCompare.lessEqual,
    List<VertexBufferLayout> vertexBuffers = const [],
    List<RenderBindingLayout> bindings = const [],
  }) : vertexBuffers = UnmodifiableListView(List.of(vertexBuffers)),
       bindings = UnmodifiableListView(List.of(bindings)) {
    validate();
  }
  final String code, vertexEntry, fragmentEntry;
  final TextureFormat colorFormat;
  final RenderBlendMode blend;
  final PrimitiveTopology topology;
  final FrontFace frontFace;
  final CullMode cullMode;
  final bool depthEnabled, depthWrite;
  final SamplerCompare depthCompare;
  final List<VertexBufferLayout> vertexBuffers;
  final List<RenderBindingLayout> bindings;
  void validate() {
    if (code.isEmpty ||
        code.length > 1048576 ||
        code.contains('\u0000') ||
        utf8.encode(code).length > 1048576 ||
        !_identifier(vertexEntry) ||
        !_identifier(fragmentEntry) ||
        !_colorFormat(colorFormat) ||
        (!depthEnabled && depthWrite) ||
        vertexBuffers.length > 8 ||
        bindings.length > 32)
      throw ArgumentError('Invalid render pipeline descriptor');
    final locations = <int>{};
    var attributes = 0;
    for (final layout in vertexBuffers) {
      layout.validate();
      attributes += layout.attributes.length;
      for (final a in layout.attributes) {
        if (!locations.add(a.shaderLocation))
          throw ArgumentError('Duplicate vertex location');
      }
    }
    if (attributes > 16) throw ArgumentError('Too many vertex attributes');
    final slots = <int>{};
    for (final binding in bindings) {
      binding.validate();
      if (!slots.add(binding.slot))
        throw ArgumentError('Duplicate render binding');
    }
  }

  List<int> encode() {
    validate();
    final words = <int>[
      1,
      colorFormat.wireCode,
      blend.wireCode,
      topology.wireCode,
      frontFace.wireCode,
      cullMode.wireCode,
      depthEnabled ? 1 : 0,
      depthWrite ? 1 : 0,
      depthCompare.wireCode,
      vertexBuffers.length,
      bindings.length,
    ];
    for (final layout in vertexBuffers) {
      words.addAll([
        layout.arrayStride,
        layout.stepMode.wireCode,
        layout.attributes.length,
      ]);
      for (final a in layout.attributes) {
        words.addAll([a.shaderLocation, a.offset, a.format.wireCode]);
      }
    }
    for (final b in bindings) {
      words.addAll([
        b.slot,
        b.kind.wireCode,
        b.visibility,
        b.viewDimension.wireCode,
        b.minBindingSize,
      ]);
    }
    return List.unmodifiable(words);
  }

  static bool _identifier(String value) =>
      value.isNotEmpty &&
      value.length <= 64 &&
      RegExp(r'^[A-Za-z_][A-Za-z0-9_]*$').hasMatch(value);
  static bool _colorFormat(TextureFormat format) => switch (format) {
    TextureFormat.rgba8Unorm ||
    TextureFormat.bgra8Unorm ||
    TextureFormat.rgba16Float ||
    TextureFormat.rgba8UnormSrgb ||
    TextureFormat.bgra8UnormSrgb => true,
    _ => false,
  };
}

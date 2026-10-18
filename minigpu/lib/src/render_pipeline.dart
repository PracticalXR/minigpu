import 'dart:async';
import 'dart:typed_data';

import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';

import 'buffer.dart';
import 'gpu_texture.dart';
import 'resource_retirement.dart';

/// Monotonic cache evidence for one retained render pipeline.
final class GpuRenderBindingCacheSnapshot {
  const GpuRenderBindingCacheSnapshot({
    required this.created,
    required this.reused,
    required this.retired,
    required this.live,
  });

  final int created, reused, retired, live;

  Map<String, int> toJson() => {
    'created': created,
    'reused': reused,
    'retired': retired,
    'live': live,
  };
}

final class GpuRenderPipeline {
  /// A supplied [retirement] replaces the one built from [owner]; it must
  /// destroy [_platform] and must not reference this pipeline, or the
  /// finalizer can never run.
  GpuRenderPipeline(
    this._platform,
    this.descriptor, [
    void Function()? owner,
    GpuResourceRetirement? retirement,
  ]) : _retirement =
           retirement ?? GpuResourceRetirement(_platform.destroy, owner) {
    _finalizer.attach(this, _retirement, detach: this);
  }
  final PlatformRenderPipeline _platform;
  final RenderPipelineDescriptor descriptor;
  final GpuResourceRetirement _retirement;
  bool _valid = true;
  bool _ready = false;
  Future<void>? _preparation;
  static final _finalizer = Finalizer<GpuResourceRetirement>(
    (token) => token.retire(),
  );
  bool get isValid => _valid && !_retirement.isRetired;

  /// Null when the backend lacks retained-binding diagnostics or is destroyed.
  GpuRenderBindingCacheSnapshot? get bindingCacheSnapshot {
    final stats = _platform.bindingCacheStats;
    if (stats == null || stats.length != 4 || stats.any((value) => value < 0)) {
      return null;
    }
    return GpuRenderBindingCacheSnapshot(
      created: stats[0],
      reused: stats[1],
      retired: stats[2],
      live: stats[3],
    );
  }

  Future<void> prepare() => _preparation ??= _prepare();

  Future<void> _prepare() async {
    final stopwatch = Stopwatch()..start();
    const limit = Duration(seconds: 30);
    // A backend that signals the settle is awaited: no wakeups while a slow
    // compile runs, and no lag behind it. Destroying the pipeline settles it
    // too. The status below still decides the outcome.
    final settled = _platform.status == 0 ? _platform.settled : null;
    if (settled != null) {
      try {
        await settled.timeout(limit);
      } on TimeoutException {
        // Reported below, by the same check a polled compile meets.
      }
    }
    // Without the signal, poll: soon at first (cache hits finish within a
    // millisecond or two), then back off to a 16 ms period for slow or stuck
    // compiles.
    var pollMs = 1;
    while (isValid) {
      switch (_platform.status) {
        case 1:
          _ready = true;
          return;
        case 2:
          throw StateError(_platform.diagnostic);
        case 3:
          throw StateError('Render pipeline was closed');
        case 0:
          break;
        default:
          throw StateError('Unknown render pipeline status');
      }
      if (stopwatch.elapsed >= limit) {
        destroy();
        throw TimeoutException('Render pipeline preparation timed out');
      }
      await Future<void>.delayed(Duration(milliseconds: pollMs));
      if (pollMs < 16) pollMs *= 2;
    }
    throw StateError('Render pipeline was destroyed during preparation');
  }

  GpuRenderPass createPass({
    required GpuTextureView color,
    GpuTextureView? depth,
    List<double> clearColor = const [0, 0, 0, 0],
    double clearDepth = 1,
    bool loadColor = false,
    bool loadDepth = false,
  }) {
    if (!isValid || !_ready) throw StateError('Render pipeline is not ready');
    if (clearColor.length != 4 ||
        clearColor.any((v) => !_finiteF32(v)) ||
        !clearDepth.isFinite ||
        clearDepth < 0 ||
        clearDepth > 1)
      throw ArgumentError('Invalid render clear values');
    _attachment(color, descriptor.colorFormat);
    if (descriptor.depthEnabled != (depth != null))
      throw ArgumentError('Depth attachment does not match pipeline');
    if (depth != null) _attachment(depth, TextureFormat.depth32Float);
    if (depth != null &&
        (_mipWidth(color) != _mipWidth(depth) ||
            _mipHeight(color) != _mipHeight(depth))) {
      throw ArgumentError('Render attachments have different dimensions');
    }
    if (loadDepth && depth == null)
      throw ArgumentError('Cannot load an absent depth attachment');
    return GpuRenderPass._(this, color, depth, [
      ...clearColor,
      clearDepth,
    ], (loadColor ? 1 : 0) | (loadDepth ? 2 : 0));
  }

  void _attachment(GpuTextureView view, TextureFormat format) {
    if (!view.isValid ||
        view.texture.descriptor.format != format ||
        view.texture.descriptor.usage & TextureUsage.renderAttachment == 0 ||
        view.descriptor.dimension != TextureViewDimension.d2 ||
        view.descriptor.mipLevelCount != 1 ||
        view.descriptor.arrayLayerCount != 1)
      throw ArgumentError('Invalid render attachment');
  }

  int _mipWidth(GpuTextureView view) {
    final value = view.texture.descriptor.width >> view.descriptor.baseMipLevel;
    return value < 1 ? 1 : value;
  }

  int _mipHeight(GpuTextureView view) {
    final value =
        view.texture.descriptor.height >> view.descriptor.baseMipLevel;
    return value < 1 ? 1 : value;
  }

  void destroy() {
    if (!_valid) return;
    _valid = false;
    _finalizer.detach(this);
    _retirement.retire();
  }
}

final class GpuRenderPass {
  GpuRenderPass._(
    this.pipeline,
    this.color,
    this.depth,
    this._clear,
    this._loadFlags,
  );
  final GpuRenderPipeline pipeline;
  final GpuTextureView color;
  final GpuTextureView? depth;
  final List<double> _clear;
  final int _loadFlags;
  final _buffers = <Buffer>[];
  final _views = <GpuTextureView>[];
  final _samplers = <GpuSampler>[];
  final _commands = <int>[];
  final _vertexSizes = <int, int>{};
  final _boundSlots = <int>{};
  (IndexFormat, int)? _index;
  bool _consumed = false;
  int _draws = 0;
  int _commandCount = 0;

  void _check() {
    if (_consumed || !pipeline.isValid || !pipeline._ready) {
      throw StateError('Render pass is consumed or pipeline is unavailable');
    }
  }

  void _admit(int words, {bool draw = false}) {
    _check();
    if (_commandCount >= 4096 ||
        _commands.length > 65536 - words ||
        (draw && _draws >= 1024)) {
      throw StateError('Render pass command budget exceeded');
    }
  }

  void _append(List<int> words, {bool draw = false}) {
    _commands.addAll(words);
    _commandCount++;
    if (draw) _draws++;
  }

  int _reference<T>(List<T> references, T value) {
    final index = references.indexOf(value);
    if (index >= 0) return index;
    if (references.length >= 256)
      throw StateError('Render resource budget exceeded');
    references.add(value);
    return references.length - 1;
  }

  void setVertexBuffer(
    int slot,
    Buffer buffer, {
    int offset = 0,
    int? sizeBytes,
  }) {
    _admit(5);
    if (slot < 0 || slot >= pipeline.descriptor.vertexBuffers.length) {
      throw ArgumentError('Unknown vertex buffer slot');
    }
    final size = _range(buffer, offset, sizeBytes, BufferUsage.vertex);
    final index = _reference(_buffers, buffer);
    _append([1, slot, index, offset, size]);
    _vertexSizes[slot] = size;
  }

  void setIndexBuffer(
    Buffer buffer,
    IndexFormat format, {
    int offset = 0,
    int? sizeBytes,
  }) {
    _admit(5);
    final size = _range(
      buffer,
      offset,
      sizeBytes,
      BufferUsage.index,
      alignment: format.bytes,
    );
    final index = _reference(_buffers, buffer);
    _append([2, index, offset, size, format.wireCode]);
    _index = (format, size);
  }

  RenderBindingLayout _binding(int slot) {
    for (final binding in pipeline.descriptor.bindings) {
      if (binding.slot == slot) return binding;
    }
    throw ArgumentError('Unknown render binding');
  }

  void setBuffer(int binding, Buffer buffer, {int offset = 0, int? sizeBytes}) {
    _admit(5);
    final layout = _binding(binding);
    final usage = switch (layout.kind) {
      RenderBindingKind.uniformBuffer => BufferUsage.uniform,
      RenderBindingKind.readOnlyStorageBuffer => BufferUsage.storage,
      _ => throw ArgumentError('Binding does not accept a buffer'),
    };
    final size = _range(buffer, offset, sizeBytes, usage);
    if (size < layout.minBindingSize)
      throw ArgumentError('Buffer range is too small');
    final index = _reference(_buffers, buffer);
    _append([3, binding, index, offset, size]);
    _boundSlots.add(binding);
  }

  void setTexture(int binding, GpuTextureView view) {
    _admit(3);
    final layout = _binding(binding);
    if (!view.isValid) throw StateError('Texture view is destroyed');
    final source = view.texture.descriptor;
    final validKind = switch (layout.kind) {
      RenderBindingKind.sampledFloat =>
        !source.format.isDepth && source.format.isFilterable,
      RenderBindingKind.sampledUnfilterableFloat => !source.format.isDepth,
      RenderBindingKind.sampledDepth => source.format.isDepth,
      _ => false,
    };
    if (!validKind ||
        source.usage & TextureUsage.textureBinding == 0 ||
        view.descriptor.dimension != layout.viewDimension ||
        identical(view.texture, color.texture) ||
        identical(view.texture, depth?.texture)) {
      throw ArgumentError(
        'Texture binding is incompatible or aliases an attachment',
      );
    }
    final index = _reference(_views, view);
    _append([4, binding, index]);
    _boundSlots.add(binding);
  }

  void setSampler(int binding, GpuSampler sampler) {
    _admit(3);
    if (!sampler.isValid) throw StateError('Sampler is destroyed');
    final descriptor = sampler.descriptor;
    final valid = switch (_binding(binding).kind) {
      RenderBindingKind.filteringSampler => descriptor.compare == null,
      RenderBindingKind.nonFilteringSampler =>
        descriptor.compare == null &&
            descriptor.magFilter == SamplerFilter.nearest &&
            descriptor.minFilter == SamplerFilter.nearest &&
            descriptor.mipmapFilter == SamplerFilter.nearest,
      RenderBindingKind.comparisonSampler => descriptor.compare != null,
      _ => false,
    };
    if (!valid) throw ArgumentError('Sampler does not match binding layout');
    final index = _reference(_samplers, sampler);
    _append([5, binding, index]);
    _boundSlots.add(binding);
  }

  void draw(
    int vertexCount, {
    int instanceCount = 1,
    int firstVertex = 0,
    int firstInstance = 0,
  }) {
    _draw(vertexCount, instanceCount, firstVertex, firstInstance, false, 0);
  }

  void drawIndexed(
    int indexCount, {
    int instanceCount = 1,
    int firstIndex = 0,
    int baseVertex = 0,
    int firstInstance = 0,
  }) {
    _draw(
      indexCount,
      instanceCount,
      firstIndex,
      firstInstance,
      true,
      baseVertex,
    );
  }

  void _draw(
    int count,
    int instances,
    int first,
    int firstInstance,
    bool indexed,
    int base,
  ) {
    _admit(indexed ? 6 : 5, draw: true);
    if (count < 1 ||
        count > 16777216 ||
        instances < 1 ||
        instances > 1048576 ||
        !_u32(first) ||
        !_u32(firstInstance) ||
        count - 1 > 0xffffffff - first ||
        instances - 1 > 0xffffffff - firstInstance ||
        base < -2147483648 ||
        base > 2147483647) {
      throw ArgumentError('Draw count or range is invalid');
    }
    for (final binding in pipeline.descriptor.bindings) {
      if (!_boundSlots.contains(binding.slot))
        throw StateError('Draw is missing a shader binding');
    }
    if (indexed) {
      final index = _index;
      if (index == null) throw StateError('Draw is missing an index buffer');
      final capacity = index.$2 ~/ index.$1.bytes;
      if (first > capacity || count > capacity - first)
        throw ArgumentError('Index range exceeds buffer');
    }
    for (
      var slot = 0;
      slot < pipeline.descriptor.vertexBuffers.length;
      slot++
    ) {
      final layout = pipeline.descriptor.vertexBuffers[slot];
      final size = _vertexSizes[slot];
      if (size == null) throw StateError('Draw is missing a vertex buffer');
      var lastAttribute = 0;
      for (final attribute in layout.attributes) {
        final end = attribute.offset + attribute.format.bytes;
        if (end > lastAttribute) lastAttribute = end;
      }
      // Indexed values stay on the GPU and use WebGPU robust access rules.
      final last = layout.stepMode == VertexStepMode.instance
          ? firstInstance + instances - 1
          : indexed
          ? 0
          : first + count - 1;
      if (lastAttribute > size ||
          last > (size - lastAttribute) ~/ layout.arrayStride) {
        throw ArgumentError('Draw range exceeds vertex or instance buffer');
      }
    }
    _append(
      indexed
          ? [7, count, instances, first, base & 0xffffffff, firstInstance]
          : [6, count, instances, first, firstInstance],
      draw: true,
    );
  }

  void setViewport(
    double x,
    double y,
    double width,
    double height, {
    double minDepth = 0,
    double maxDepth = 1,
  }) {
    _admit(7);
    final values = Float32List.fromList([
      x,
      y,
      width,
      height,
      minDepth,
      maxDepth,
    ]);
    if (values.any((v) => !v.isFinite) ||
        values[0] < 0 ||
        values[1] < 0 ||
        values[2] <= 0 ||
        values[3] <= 0 ||
        values[0] > _width(color) - values[2] ||
        values[1] > _height(color) - values[3] ||
        values[4] < 0 ||
        values[5] > 1 ||
        values[4] > values[5]) {
      throw ArgumentError('Viewport is outside attachment or depth range');
    }
    _append([8, ...values.buffer.asUint32List()]);
  }

  void setScissorRect(int x, int y, int width, int height) {
    _admit(5);
    if (x < 0 ||
        y < 0 ||
        width < 1 ||
        height < 1 ||
        width > _width(color) ||
        height > _height(color) ||
        x > _width(color) - width ||
        y > _height(color) - height) {
      throw ArgumentError('Scissor is outside attachment');
    }
    _append([9, x, y, width, height]);
  }

  /// Enqueues once. Await the device's submitted-work fence for GPU completion.
  Future<void> submit() async {
    _check();
    if (_buffers.any((b) => !b.isValid) ||
        _views.any((v) => !v.isValid) ||
        _samplers.any((s) => !s.isValid) ||
        !color.isValid ||
        (depth != null && !depth!.isValid)) {
      throw StateError('A recorded render resource was destroyed');
    }
    if (pipeline._platform.status != 1)
      throw StateError('Render pipeline is no longer ready');
    final buffers = [for (final b in _buffers) b.platformBuffer!];
    final views = [for (final v in _views) v.platformView];
    final samplers = [for (final s in _samplers) s.platformSampler];
    _consumed = true;
    try {
      await pipeline._platform.submitRender(
        color: color.platformView,
        depth: depth?.platformView,
        clear: _clear,
        loadFlags: _loadFlags,
        buffers: buffers,
        views: views,
        samplers: samplers,
        commands: List.unmodifiable(_commands),
      );
    } finally {
      _clearCommands();
    }
  }

  void discard() {
    if (_consumed) return;
    _consumed = true;
    _clearCommands();
  }

  void _clearCommands() {
    _buffers.clear();
    _views.clear();
    _samplers.clear();
    _commands.clear();
    _vertexSizes.clear();
    _boundSlots.clear();
    _index = null;
  }

  int _range(
    Buffer buffer,
    int offset,
    int? requested,
    int usage, {
    int alignment = 4,
  }) {
    if (!buffer.isValid) throw StateError('Buffer is destroyed');
    final descriptor = buffer.descriptor;
    if (descriptor == null ||
        descriptor.usage & usage == 0 ||
        offset < 0 ||
        offset > descriptor.sizeBytes ||
        offset % alignment != 0) {
      throw ArgumentError('Buffer usage or offset is invalid');
    }
    final size = requested ?? descriptor.sizeBytes - offset;
    if (size < 1 ||
        size % alignment != 0 ||
        size > descriptor.sizeBytes - offset) {
      throw ArgumentError('Buffer range is invalid');
    }
    return size;
  }
}

int _width(GpuTextureView view) =>
    _mip(view.texture.descriptor.width, view.descriptor.baseMipLevel);
int _height(GpuTextureView view) =>
    _mip(view.texture.descriptor.height, view.descriptor.baseMipLevel);
int _mip(int extent, int level) {
  final value = extent >> level;
  return value < 1 ? 1 : value;
}

bool _u32(int value) => value >= 0 && value <= 0xffffffff;
bool _finiteF32(double value) =>
    value.isFinite && value.abs() <= 3.4028234663852886e38;

library;

enum TextureFormat {
  rgba8Unorm(1, 4, false, true, true),
  bgra8Unorm(2, 4, false, true, false),
  rgba16Float(3, 8, false, true, true),
  r32Float(4, 4, false, false, true),
  depth32Float(5, 4, true, false, false),
  rgba8UnormSrgb(6, 4, false, true, false),
  bgra8UnormSrgb(7, 4, false, true, false);

  const TextureFormat(
    this.wireCode,
    this.bytesPerTexel,
    this.isDepth,
    this.isFilterable,
    this.supportsStorage,
  );
  final int wireCode;
  final int bytesPerTexel;
  final bool isDepth;
  final bool isFilterable;
  final bool supportsStorage;
}

abstract final class TextureUsage {
  static const copySrc = 1,
      copyDst = 2,
      textureBinding = 4,
      storageBinding = 8,
      renderAttachment = 16,
      known = 31;
}

final class TextureDescriptor {
  TextureDescriptor({
    required this.width,
    required this.height,
    required this.format,
    required this.usage,
    this.arrayLayerCount = 1,
    this.mipLevelCount = 1,
  }) {
    validate();
  }
  final int width, height, usage, arrayLayerCount, mipLevelCount;
  final TextureFormat format;
  void validate() {
    final maxMip = _maxMip(width > height ? width : height);
    if (width < 1 ||
        width > 32768 ||
        height < 1 ||
        height > 32768 ||
        arrayLayerCount < 1 ||
        arrayLayerCount > 32768 ||
        mipLevelCount < 1 ||
        mipLevelCount > maxMip ||
        usage == 0 ||
        usage & ~TextureUsage.known != 0 ||
        (format.isDepth &&
            usage &
                    ~(TextureUsage.textureBinding |
                        TextureUsage.renderAttachment) !=
                0) ||
        (usage & TextureUsage.storageBinding != 0 && !format.supportsStorage))
      throw ArgumentError('Invalid texture descriptor');
  }

  static int _maxMip(int extent) {
    var count = 1;
    while (extent > 1) {
      extent >>= 1;
      count++;
    }
    return count;
  }
}

enum TextureViewDimension {
  d2(0),
  d2Array(1),
  cube(2),
  cubeArray(3);

  const TextureViewDimension(this.wireCode);
  final int wireCode;
}

final class TextureViewDescriptor {
  TextureViewDescriptor({
    this.dimension = TextureViewDimension.d2,
    this.baseMipLevel = 0,
    this.mipLevelCount = 1,
    this.baseArrayLayer = 0,
    this.arrayLayerCount = 1,
  }) {
    validate();
  }
  final TextureViewDimension dimension;
  final int baseMipLevel, mipLevelCount, baseArrayLayer, arrayLayerCount;
  void validate() {
    if (baseMipLevel < 0 ||
        baseArrayLayer < 0 ||
        mipLevelCount < 1 ||
        arrayLayerCount < 1) {
      throw ArgumentError('Invalid texture view descriptor');
    }
  }

  void validateFor(TextureDescriptor texture) {
    validate();
    if (baseMipLevel > texture.mipLevelCount ||
        mipLevelCount > texture.mipLevelCount - baseMipLevel ||
        baseArrayLayer > texture.arrayLayerCount ||
        arrayLayerCount > texture.arrayLayerCount - baseArrayLayer ||
        (dimension == TextureViewDimension.d2 && arrayLayerCount != 1) ||
        ((dimension == TextureViewDimension.cube ||
                dimension == TextureViewDimension.cubeArray) &&
            (texture.width != texture.height ||
                arrayLayerCount % 6 != 0 ||
                (dimension == TextureViewDimension.cube &&
                    arrayLayerCount != 6)))) {
      throw ArgumentError('Texture view exceeds source texture');
    }
  }
}

enum SamplerAddressMode {
  clampToEdge(0),
  repeat(1),
  mirrorRepeat(2);

  const SamplerAddressMode(this.wireCode);
  final int wireCode;
}

enum SamplerFilter {
  nearest(0),
  linear(1);

  const SamplerFilter(this.wireCode);
  final int wireCode;
}

enum SamplerCompare {
  never(1),
  less(2),
  equal(3),
  lessEqual(4),
  greater(5),
  notEqual(6),
  greaterEqual(7),
  always(8);

  const SamplerCompare(this.wireCode);
  final int wireCode;
}

final class SamplerDescriptor {
  SamplerDescriptor({
    this.addressModeU = SamplerAddressMode.clampToEdge,
    this.addressModeV = SamplerAddressMode.clampToEdge,
    this.addressModeW = SamplerAddressMode.clampToEdge,
    this.magFilter = SamplerFilter.nearest,
    this.minFilter = SamplerFilter.nearest,
    this.mipmapFilter = SamplerFilter.nearest,
    this.lodMinClamp = 0,
    this.lodMaxClamp = 32,
    this.compare,
  }) {
    validate();
  }
  final SamplerAddressMode addressModeU, addressModeV, addressModeW;
  final SamplerFilter magFilter, minFilter, mipmapFilter;
  final double lodMinClamp, lodMaxClamp;
  final SamplerCompare? compare;
  void validate() {
    if (!lodMinClamp.isFinite ||
        !lodMaxClamp.isFinite ||
        lodMinClamp < 0 ||
        lodMinClamp > lodMaxClamp ||
        lodMaxClamp > 32)
      throw ArgumentError('Invalid sampler descriptor');
  }
}

enum TextureBindingKind {
  sampledFloat(0),
  sampledUnfilterableFloat(1),
  sampledDepth(2),
  writeOnlyStorage(3);

  const TextureBindingKind(this.wireCode);
  final int wireCode;
}

enum SamplerBindingKind {
  filtering(0),
  nonFiltering(1),
  comparison(2);

  const SamplerBindingKind(this.wireCode);
  final int wireCode;
}

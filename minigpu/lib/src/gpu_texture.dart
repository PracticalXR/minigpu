import 'dart:typed_data';
import 'package:minigpu_platform_interface/minigpu_platform_interface.dart';

import 'resource_retirement.dart';

final class GpuTexture {
  GpuTexture(
    this._texture,
    this.descriptor, [
    void Function()? owner,
    this._createView,
    GpuResourceRetirement? retirement,
  ]) : _retirement =
           retirement ?? GpuResourceRetirement(_texture.destroy, owner) {
    _finalizer.attach(this, _retirement, detach: this);
  }
  final PlatformTexture _texture;
  final TextureDescriptor descriptor;
  final GpuResourceRetirement _retirement;
  final GpuTextureView Function(
    PlatformTextureView,
    GpuTexture,
    TextureViewDescriptor,
  )?
  _createView;
  bool _valid = true;
  static final _finalizer = Finalizer<GpuResourceRetirement>((token) {
    token.retire();
  });
  bool get isValid => _valid && !_retirement.isRetired;
  GpuResourceRetirement get retirement => _retirement;

  /// Whether the loaded backend supports rectangular uploads. Older backends
  /// remain usable through [write]; this never emulates a partial upload.
  bool get supportsRegionWrite => isValid && _texture.supportsRegionWrite;
  GpuTextureView createView([TextureViewDescriptor? descriptor]) {
    _check();
    final value = descriptor ?? TextureViewDescriptor();
    value.validateFor(this.descriptor);
    final platformView = _texture.createView(value);
    return _createView?.call(platformView, this, value) ??
        GpuTextureView(platformView, this, value);
  }

  Future<void> write(Uint8List bytes, {int mipLevel = 0, int arrayLayer = 0}) {
    _check();
    if (this.descriptor.usage & TextureUsage.copyDst == 0 ||
        this.descriptor.format.isDepth ||
        mipLevel < 0 ||
        mipLevel >= this.descriptor.mipLevelCount ||
        arrayLayer < 0 ||
        arrayLayer >= this.descriptor.arrayLayerCount) {
      throw ArgumentError('Invalid texture upload');
    }
    final rawWidth = this.descriptor.width >> mipLevel;
    final rawHeight = this.descriptor.height >> mipLevel;
    final w = rawWidth < 1 ? 1 : rawWidth;
    final h = rawHeight < 1 ? 1 : rawHeight;
    final requiredBytes = w * h * this.descriptor.format.bytesPerTexel;
    if (bytes.lengthInBytes != requiredBytes) {
      throw ArgumentError('Texture upload has incorrect byte length');
    }
    return _texture.write(bytes, mipLevel: mipLevel, arrayLayer: arrayLayer);
  }

  /// Uploads a tightly packed rectangle of one mip and array layer.
  ///
  /// Completion releases this call's source bytes, not outstanding GPU work.
  /// Streamed callers must bound queued bytes and await the GPU queue fence
  /// before reusing/retiring resources or declaring pixels ready.
  Future<void> writeRegion(
    Uint8List bytes, {
    required int x,
    required int y,
    required int width,
    required int height,
    int mipLevel = 0,
    int arrayLayer = 0,
  }) {
    _check();
    if (descriptor.usage & TextureUsage.copyDst == 0 ||
        descriptor.format.isDepth ||
        mipLevel < 0 ||
        mipLevel >= descriptor.mipLevelCount ||
        arrayLayer < 0 ||
        arrayLayer >= descriptor.arrayLayerCount) {
      throw ArgumentError('Invalid texture region upload');
    }
    final w = (descriptor.width >> mipLevel).clamp(1, descriptor.width);
    final h = (descriptor.height >> mipLevel).clamp(1, descriptor.height);
    // Compare before adding: hostile integer ranges must never wrap at FFI/JS.
    if (x < 0 ||
        y < 0 ||
        x >= w ||
        y >= h ||
        width <= 0 ||
        height <= 0 ||
        width > w - x ||
        height > h - y ||
        bytes.lengthInBytes > 0x7fffffff ||
        bytes.lengthInBytes !=
            width * height * descriptor.format.bytesPerTexel) {
      throw ArgumentError('Texture region bounds or byte length are invalid');
    }
    if (!_texture.supportsRegionWrite) {
      throw UnsupportedError('Region uploads are unsupported by this backend');
    }
    return _texture.writeRegion(
      bytes,
      x: x,
      y: y,
      width: width,
      height: height,
      mipLevel: mipLevel,
      arrayLayer: arrayLayer,
    );
  }

  /// Destroys the texture. Uploads, dispatches and render passes issued
  /// before this call still run with it, and its GPU memory is freed once they
  /// have, even while a render pipeline's cached bind group still names one of
  /// its views. Later use throws [StateError].
  void destroy() {
    if (!_valid) {
      return;
    }
    _valid = false;
    _finalizer.detach(this);
    _retirement.retire();
  }

  void _check() {
    if (!isValid) {
      throw StateError('Texture is destroyed');
    }
  }
}

final class GpuTextureView {
  GpuTextureView(
    this._view,
    this.texture,
    this.descriptor, [
    void Function()? owner,
    GpuResourceRetirement? retirement,
  ]) : _retirement = retirement ?? GpuResourceRetirement(_view.destroy, owner) {
    _finalizer.attach(this, _retirement, detach: this);
  }
  final PlatformTextureView _view;
  final GpuTexture texture;
  final TextureViewDescriptor descriptor;
  final GpuResourceRetirement _retirement;
  bool _valid = true;
  static final _finalizer = Finalizer<GpuResourceRetirement>((token) {
    token.retire();
  });
  bool get isValid => _valid && texture.isValid && !_retirement.isRetired;
  GpuResourceRetirement get retirement => _retirement;
  void destroy() {
    if (!_valid) {
      return;
    }
    _valid = false;
    _finalizer.detach(this);
    _retirement.retire();
  }

  PlatformTextureView get platformView {
    if (!isValid) {
      throw StateError('Texture view is destroyed');
    }
    return _view;
  }
}

final class GpuSampler {
  GpuSampler(
    this._sampler,
    this.descriptor, [
    void Function()? owner,
    GpuResourceRetirement? retirement,
  ]) : _retirement =
           retirement ?? GpuResourceRetirement(_sampler.destroy, owner) {
    _finalizer.attach(this, _retirement, detach: this);
  }
  final PlatformSampler _sampler;
  final SamplerDescriptor descriptor;
  final GpuResourceRetirement _retirement;
  bool _valid = true;
  static final _finalizer = Finalizer<GpuResourceRetirement>((token) {
    token.retire();
  });
  bool get isValid => _valid && !_retirement.isRetired;
  GpuResourceRetirement get retirement => _retirement;
  void destroy() {
    if (!_valid) {
      return;
    }
    _valid = false;
    _finalizer.detach(this);
    _retirement.retire();
  }

  PlatformSampler get platformSampler {
    if (!isValid) {
      throw StateError('Sampler is destroyed');
    }
    return _sampler;
  }
}

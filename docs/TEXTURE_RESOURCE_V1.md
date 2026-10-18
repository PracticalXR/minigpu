# Owned textures, views and samplers v1

Primary-frozen G01 implementation contract, September 9, 2026. This is an
additive generic minigpu API. It does not add engine/room policy or change the
existing imported-video/shared-output APIs. Implementation and evidence are
recorded separately; this document alone is not a support claim.

## Deliberate v1 bounds

Support owned 2D textures, array layers, mip levels, 2D/array/cube views,
samplers and explicit compute bindings. All textures have sample count one.
Multisampling, 1D/3D textures, compressed formats, reinterpretation through
alternate view formats, indirect commands and external imports are not silently
emulated. Raster pipelines follow this resource layer.

Use the WebGPU [texture descriptor](https://gpuweb.github.io/types/interfaces/GPUTextureDescriptor.html),
[sampler descriptor](https://gpuweb.github.io/types/interfaces/GPUSamplerDescriptor.html),
[sampled binding](https://gpuweb.github.io/types/interfaces/GPUTextureBindingLayout.html)
and [storage binding](https://gpuweb.github.io/types/interfaces/GPUStorageTextureBindingLayout.html)
semantics. Public integer codes below are our stable codes, not Dawn enum ordinals.

## Dart descriptors

Immutable descriptors live in platform_interface and are re-exported by minigpu.
Constructors validate; unknown values never get a default interpretation.

- `TextureFormat`: `rgba8Unorm=1`, `bgra8Unorm=2`, `rgba16Float=3`,
  `r32Float=4`, `depth32Float=5`, `rgba8UnormSrgb=6`, `bgra8UnormSrgb=7`.
  Expose `wireCode`, `bytesPerTexel` (8 for rgba16, 4 otherwise), `isDepth`,
  `isFilterable` (false for r32/depth), `supportsStorage` (rgba8, rgba16, r32).
- `TextureUsage`: copySrc=1, copyDst=2, textureBinding=4, storageBinding=8,
  renderAttachment=16; known mask31, required nonzero. Depth supports only
  textureBinding/renderAttachment in this v1. Storage use must match format.
- `TextureDescriptor({required width, required height, required format,
  required usage, arrayLayerCount=1, mipLevelCount=1})`. Integer extents/layers
  1..32768; mip count 1..floor(log2(max(width,height)))+1. Actual device limits
  still apply. No product quota is implied by these generic upper bounds.
- `TextureViewDimension`: d2=0, d2Array=1, cube=2, cubeArray=3.
- `TextureViewDescriptor({dimension=d2, baseMipLevel=0, mipLevelCount=1,
  baseArrayLayer=0, arrayLayerCount=1})`. Nonnegative bases, positive counts,
  bounds within source descriptor. d2 requires one layer; cube requires six,
  cubeArray a positive multiple of six; cube textures must be square. No format
  override. `validateFor(TextureDescriptor)` performs source-relative validation.
- `SamplerAddressMode`: clampToEdge=0, repeat=1, mirrorRepeat=2.
- `SamplerFilter`: nearest=0, linear=1.
- `SamplerCompare`: never=1, less=2, equal=3, lessEqual=4, greater=5,
  notEqual=6, greaterEqual=7, always=8; null/ABI zero means no comparison.
- `SamplerDescriptor({addressModeU=clampToEdge, addressModeV=clampToEdge,
  addressModeW=clampToEdge, magFilter=nearest, minFilter=nearest,
  mipmapFilter=nearest, lodMinClamp=0, lodMaxClamp=32, compare})`.
  Finite clamps, 0 <= min <= max <=32. Anisotropy fixed at one.
- `TextureBindingKind`: sampledFloat=0, sampledUnfilterableFloat=1,
  sampledDepth=2, writeOnlyStorage=3. Sampled requires textureBinding usage.
  Float requires a filterable color format; unfilterable accepts color; depth
  requires depth. Storage requires storage usage/format and a one-mip d2/d2Array
  view. This is explicit layout metadata, never WGSL-text inference.
- `SamplerBindingKind`: filtering=0, nonFiltering=1, comparison=2.
  Comparison requires compare; other kinds forbid compare. Non-filtering also
  requires all three nearest filters.

## Public and platform operations

`Minigpu.createTexture(TextureDescriptor)` -> `GpuTexture`;
`Minigpu.createSampler(SamplerDescriptor)` -> `GpuSampler`.
Owned wrappers expose immutable descriptor, `isValid`, idempotent `destroy()`.
`GpuTexture.createView([TextureViewDescriptor?])` -> `GpuTextureView`, default
descriptor above. A view has its descriptor and originating texture. It cannot
outlive the public source texture's validity.

`GpuTexture.write(Uint8List bytes, {int mipLevel=0, int arrayLayer=0})` uploads
one entire mip of one layer, tightly packed in its declared format. Require
copyDst, color format, valid mip/layer and exact byte count. No conversion,
readback, vertical flip or invented alpha/color encoding. Return Future<void>
for host enqueue/upload lifetime, not a GPU completion promise.

`GpuTexture.supportsRegionWrite` reports whether the loaded backend exposes the
additive region-upload v1. It is false for a destroyed texture or an older binary
or module; the existing whole-mip method remains available.
`GpuTexture.writeRegion(Uint8List bytes, {required int x, required int y,
required int width, required int height, int mipLevel=0, int arrayLayer=0})`
uploads one positive rectangular extent within one mip of one layer. Payloads
are tightly packed in the declared format, with no stride, conversion, slicing,
flip or mip generation. Require copyDst, color, valid mip/layer, nonnegative
origin, source-relative bounds and exactly width*height*bytesPerTexel bytes.
The C ABI byte count must fit a positive signed 32-bit integer. Validate ranges
before integer narrowing or origin/extent addition.

The source payload may be reused once the returned Future completes. This is
still not a GPU completion promise. Streamed consumers must reserve and bound
their own queued/staging bytes and use `waitForSubmittedWork()` before declaring
GPU readiness, promoting a view, or retiring resources. This method copies the
requested region into backend-owned queued bytes; it is not an imported-video
zero-copy path or an automatic streaming scheduler.

`ComputeShader.setTexture(int slot, GpuTextureView view,
TextureBindingKind kind)` and `setSampler(int slot, GpuSampler sampler,
SamplerBindingKind kind)` use slots0..31. Check alive/context/layout/usage before
backend use. Keep wrapper bindings pinned and validate their liveness at dispatch
so destroying an explicitly bound texture/view/sampler refuses later dispatch.
Replacing any slot through any binding family clears the previous wrapper owner.

Use analogous platform types/methods (`PlatformTexture`, `PlatformTextureView`,
`PlatformSampler`). New methods on existing platform classes have concrete
UnsupportedError defaults, preserving old adapters. There is no legacy fallback
for texture creation or typed texture/sampler binding.

Extend the existing weak child tracking and teardown, without retaining every
temporary texture strongly. Close shaders, views, textures, samplers before
platform detach. A view pins its source; explicit source destroy invalidates its
views. Platform wrappers carry exact adapter ownership. No new default device.

## C ABI

Add `include/minigpu_resources.h` and `src/minigpu_resources.cpp`, exported by
the existing library and Emscripten module. Existing `minigpu.h` may include the
new header. All functions contain exceptions; pointers are owned C API handles,
not an interface accepting arbitrary untrusted memory. Callers must retire
children before destroying their explicit context handle.

```c
uint32_t mgpuTextureResourceVersion(void); /* 1 */
MGPUTexture* mgpuCreateTexture2D(int width, int height, int format, int usage,
                               int layers, int mipLevels);
MGPUTexture* mgpuContextCreateTexture2D(MGPUContextHandle* context,
    int width, int height, int format, int usage, int layers, int mipLevels);
MGPUTextureView* mgpuCreateTextureView(MGPUTexture* texture, int dimension,
    int baseMip, int mipCount, int baseLayer, int layerCount);
MGPUSampler* mgpuCreateSampler(int addressU, int addressV, int addressW,
    int mag, int min, int mip, float lodMin, float lodMax, int compare);
MGPUSampler* mgpuContextCreateSampler(MGPUContextHandle* context,
    int addressU, int addressV, int addressW, int mag, int min, int mip,
    float lodMin, float lodMax, int compare);
int mgpuWriteTexture2D(MGPUTexture* texture, const uint8_t* bytes,
                     int byteCount, int mipLevel, int arrayLayer);
uint32_t mgpuTextureRegionUploadVersion(void); /* additive region ABI: 1 */
int mgpuWriteTextureRegion2D(MGPUTexture* texture, const uint8_t* bytes,
    int byteCount, int x, int y, int width, int height, int mipLevel,
    int arrayLayer);
int mgpuSetOwnedTexture(MGPUComputeShader* shader, int slot,
                       MGPUTextureView* view, int kind);
int mgpuSetOwnedSampler(MGPUComputeShader* shader, int slot,
                       MGPUSampler* sampler, int kind);
void mgpuDestroyTexture(MGPUTexture* texture);
void mgpuDestroyTextureView(MGPUTextureView* view);
void mgpuDestroySampler(MGPUSampler* sampler);
```

Creation returns null on invalid/unsupported/unavailable/allocation failure;
Dart validates descriptors first then reports a StateError on null. Status
operations:0 accepted,1 malformed/usage/layout/foreign-owner,2 unavailable or
stale device,3 backend failure. Absence of the version/export is UnsupportedError.
Web resolves exports only through MinigpuModule; no eval or global Module probe.
The original texture resource version stays 1. The optional region-version
export is queried separately, and missing region exports refuse with
UnsupportedError without disabling whole-mip uploads. Both native and browser
use the same C++ validation and Queue.WriteTexture implementation. Accepted
region uploads retain the exact creation device and backing state; deferred
work rechecks liveness/device identity and flushes earlier compute in order.

Native creation/bind/upload must use an already ready context, never getDevice's
lazy initialization. Add a retained ready-device accessor on MGPU that does not
start/reset a device; compare against retained creation-device identity at use.
Retaining that identity prevents recycled pointer equality. Existing compute
shaders record their creation device for this check; preserve legacy behavior.
Query actual texture dimension/layer/binding limits. Serialize GPU mutations
through the existing queue/mutex and flush pending compute before a texture upload.
Validate again before deferred execution; no stale resource can target a new device.

Native texture/view/sampler backing states use retained WebGPU references and
shared ownership between queued work/bindings/views. Releasing a public C handle
must not free a backing view still referenced by accepted queued work. Retire
references on slot replacement and shader cleanup. Do not eagerly destroy an
underlying texture before prior queued work has consumed it. A bounded host
already owns admission; this is not permission for guests to bypass quotas.

Extend the unified compute binding table with typed sampled/storage views and
samplers. Layout kind, dimension, format or sampler type changes invalidate the
pipeline layout; resource changes invalidate bind groups. Legacy video texture
binding remains its existing unfilterable-float d2 behavior and drops any prior
owned binding at that slot. A typed bind followed by a legacy buffer bind must
restore the correct buffer layout.

## Assigned acceptance

Root owns coordinated checks after native and Dart integration: actual owned
RGBA upload, sampled compute read, storage texture write/read, nearest sampler,
view bounds, wrong-usage/kind/foreign-context refusal and disposal. The same
backend source supplies native and browser. Use the existing cached build and
temporary browser harness once after integration; no independent regressions.
Record generation/lifecycle coverage precisely, not as full device-loss recovery.
# Implementation receipt — September 9, 2026

The public Dart API, native FFI (default/secondary context), scoped browser
adapter and C++ WebGPU implementation are implemented. Root reviewed descriptors,
ownership, slot replacement, partial-allocation cleanup and context retirement.

The native C ABI scenario passed on Windows/Dawn/D3D11, RTX 3090: RGBA upload,
nearest sampling into a storage texture and compute readback, invalid view/kind/
usage refusal, destroyed-source dispatch refusal and same-versus-foreign-context
binding. Valid creation before device initialization refuses without opening it.
The public Dart/Wasm browser scenario passed in isolated headless Chrome against
the staged Emscripten candidate, including thin-mip upload and slot replacement.
No browser exceptions were reported. Five owning-wrapper lifecycle checks passed,
including texture-family retirement and once-only shared finalizer tokens.

Weak registrations retain cleanup tokens even after a wrapper becomes unreachable.
Context closure retires those tokens before detaching its backend; a delayed
finalizer cannot subsequently destroy through an already-deleted context.

These are compute/resource checks, not raster, mobile, visible presentation,
device-loss recovery or performance qualification. Staged native/web candidates
have not been installed into a deployed application.

## Additive region-upload receipt - September 26, 2026

The optional region version/export, public and platform methods, native FFI and
scoped browser bindings are implemented. Whole-mip upload delegates to the same
validated C++ path and the original texture-resource ABI stays version 1.

The focused Windows/Dawn/D3D11 RTX 3090 C ABI scenario passes: nonzero origin,
narrow multi-row upload without padded payload, source reuse after enqueue,
unchanged neighboring texels and other array layers, selected/one-texel mips,
and malformed bounds/length/mip/layer/usage/depth refusal. The public Dart/Wasm
scenario also passes in headless Chrome against a temporary staged backend,
including a sliced byte view and destroyed-resource refusal. There are no page
errors or GPU validation messages. Initial browser harness failures were its
module MIME type; the corrected harness exercises actual GPU compute/readback.
Targeted analysis has no new errors or warnings; existing adapter warnings remain.
One read-only review found no material issue in the upload/consumer boundaries.

These receipts qualify resource uploads, not physical display, device-loss
recovery, mobile, deployed applications or a complete streamed asset format.
Older-binary fallback and mid-upload cancellation are inspected but not exercised
by the new focused scenarios. Build/test artifacts stay in system temp under
mgpu-texture-region-*. The cached browser build destination was restored after
the staged check; packaged web artifacts were not replaced.

## Current-device texture limits - September 26, 2026

`Minigpu.currentDeviceTextureLimits` returns an optional snapshot of the owned
device generation, maximum 2D dimension, maximum array layers and region-upload
support. Native queries the retained ready device; web reads the exact owned
GPUDevice. Missing optional exports on older libraries return unavailable rather
than guessed limits. `ownsTexture` also checks the creation epoch, so a wrapper
from an earlier device cannot be promoted by a new owner.

The cached browser backend is now rebuilt into the package's normal web assets.
A subsequent Dart/Wasm Chrome check against those packaged binaries passes large
cross-row sRGB and small linear-data uploads, sampling and readback with no page
errors or GPU validation messages. The native counterpart passes on Windows/Dawn.
This supersedes the staged-only packaging status above; mobile, driver-memory
measurement, physical display and device-loss recovery remain unqualified.

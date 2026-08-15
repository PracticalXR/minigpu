/// PLANAR / SUBSAMPLED HOST FRAMES → RGBA, ON THE GPU.
///
/// Cameras and screen capture hand out NV12, I420 or YUY2 far more often than
/// RGBA — RGBA usually only appears because someone asked a converter for it.
/// When those arrive in host memory the obvious move is a per-pixel Dart loop,
/// and that is the single worst thing to put on a display path: it is O(pixels)
/// on the isolate that builds the UI, it runs inside the capture callback, and
/// at 1080p60 it is more than a frame interval of work. It does not show up as
/// a slower preview, it shows up as the app freezing.
///
/// So the planes are uploaded as raw bytes and unpacked by a compute shader.
/// The host does one memcpy-shaped upload; the arithmetic happens where there
/// are thousands of lanes for it.
///
/// ── COLOUR ──────────────────────────────────────────────────────────────────
/// BT.709, LIMITED range (luma 16..235, chroma 16..240) — what Media
/// Foundation and Windows.Graphics.Capture deliver for camera and screen NV12.
/// Getting this wrong is not subtle but it IS silent: full-range content
/// decoded as limited looks washed out, and anything measuring quality against
/// an RGB reference then reports a loss that has nothing to do with the codec.
/// [PlanarColorRange] exists so a caller who knows better can say so.
///
/// ── CHROMA SITING ───────────────────────────────────────────────────────────
/// Point-sampled at (x>>1, y>>1). Bilinear chroma is marginally nicer on
/// gradients and costs a sampler plus four fetches per pixel on a path that
/// runs per displayed frame; for preview it does not survive the trip.
library;

/// Host-memory layouts this can unpack.
enum PlanarFormat {
  /// Y plane, then interleaved UV at half resolution in both axes.
  nv12,

  /// Y, then U, then V, each at half resolution in both axes.
  i420,

  /// Packed 4:2:2, two pixels per 4 bytes: Y0 U Y1 V.
  yuy2,

  /// Packed 8-bit BGRA. No subsampling — a channel swizzle only.
  bgra8,
}

/// Whether luma/chroma use the studio-swing subset or the full 0..255 range.
enum PlanarColorRange { limited, full }

/// One plane of a host-memory frame.
class PlanarPlane {
  const PlanarPlane({
    required this.bytes,
    required this.strideBytes,
    required this.height,
  });

  final List<int> bytes;

  /// Row pitch. Capture APIs pad rows, so this is frequently NOT
  /// `width * bytesPerPixel` — assuming it is produces a picture that shears
  /// progressively down the frame, which is easy to misread as a codec fault.
  final int strideBytes;

  final int height;

  int get byteLength => strideBytes * height;
}

/// The conversion kernel.
///
/// Reads the packed planes out of a storage buffer as bytes and writes packed
/// RGBA8 (one u32 per pixel, R in the low byte) into another. Layout constants
/// arrive in `cfg` so one pipeline serves every format and size — recompiling
/// per resolution would cost seconds on the D3D11/FXC path.
///
/// cfg: [width, height, format, yOffset, yStride, uOffset, uStride, vOffset,
///       vStride, fullRange]
const String kPlanarToRgbaWgsl = '''
@group(0) @binding(0) var<storage, read_write> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<storage, read_write> cfg: array<u32>;

fn byteAt(i: u32) -> u32 {
  let w = src[i >> 2u];
  return (w >> ((i & 3u) * 8u)) & 0xFFu;
}

fn pack(r: f32, g: f32, b: f32) -> u32 {
  let q = clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0)) * 255.0
          + vec3<f32>(0.5);
  return u32(q.x) | (u32(q.y) << 8u) | (u32(q.z) << 16u) | (255u << 24u);
}

fn yuvToRgb(yRaw: f32, uRaw: f32, vRaw: f32, fullRange: u32) -> u32 {
  var y = yRaw;
  var u = uRaw - 0.5;
  var v = vRaw - 0.5;
  if (fullRange == 0u) {
    // Studio swing: undo the 16..235 / 16..240 offsets before the matrix.
    y = (y - 0.0625) * 1.164383;
    u = u * 1.138393;
    v = v * 1.138393;
  }
  // BT.709.
  return pack(y + 1.792741 * v,
              y - 0.213249 * u - 0.532909 * v,
              y + 2.112402 * u);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let width = cfg[0];
  let height = cfg[1];
  if (gid.x >= width || gid.y >= height) { return; }

  let format = cfg[2];
  let yOffset = cfg[3];
  let yStride = cfg[4];
  let uOffset = cfg[5];
  let uStride = cfg[6];
  let vOffset = cfg[7];
  let vStride = cfg[8];
  let fullRange = cfg[9];
  let outIdx = gid.y * width + gid.x;

  // 0 = nv12
  if (format == 0u) {
    let y = f32(byteAt(yOffset + gid.y * yStride + gid.x)) / 255.0;
    let c = uOffset + (gid.y >> 1u) * uStride + (gid.x >> 1u) * 2u;
    let u = f32(byteAt(c)) / 255.0;
    let v = f32(byteAt(c + 1u)) / 255.0;
    dst[outIdx] = yuvToRgb(y, u, v, fullRange);
    return;
  }

  // 1 = i420
  if (format == 1u) {
    let y = f32(byteAt(yOffset + gid.y * yStride + gid.x)) / 255.0;
    let u = f32(byteAt(uOffset + (gid.y >> 1u) * uStride + (gid.x >> 1u))) / 255.0;
    let v = f32(byteAt(vOffset + (gid.y >> 1u) * vStride + (gid.x >> 1u))) / 255.0;
    dst[outIdx] = yuvToRgb(y, u, v, fullRange);
    return;
  }

  // 2 = yuy2 — Y0 U Y1 V per pixel PAIR, so chroma comes from the even pixel.
  if (format == 2u) {
    let pair = gid.x >> 1u;
    let base = yOffset + gid.y * yStride + pair * 4u;
    let yByte = select(base + 2u, base, (gid.x & 1u) == 0u);
    let y = f32(byteAt(yByte)) / 255.0;
    let u = f32(byteAt(base + 1u)) / 255.0;
    let v = f32(byteAt(base + 3u)) / 255.0;
    dst[outIdx] = yuvToRgb(y, u, v, fullRange);
    return;
  }

  // 3 = bgra8 — swizzle only, no colour conversion.
  let p = yOffset + gid.y * yStride + gid.x * 4u;
  let b = f32(byteAt(p)) / 255.0;
  let g = f32(byteAt(p + 1u)) / 255.0;
  let r = f32(byteAt(p + 2u)) / 255.0;
  dst[outIdx] = pack(r, g, b);
}
''';

/// Number of `cfg` words the kernel reads.
const int kPlanarCfgWords = 10;

extension PlanarFormatCode on PlanarFormat {
  /// Value the kernel switches on. Must match the comments in
  /// [kPlanarToRgbaWgsl] — they are one definition in two places, so change
  /// them together.
  int get shaderCode => switch (this) {
        PlanarFormat.nv12 => 0,
        PlanarFormat.i420 => 1,
        PlanarFormat.yuy2 => 2,
        PlanarFormat.bgra8 => 3,
      };

  /// Planes the format expects, for validating what a caller handed over.
  int get planeCount => switch (this) {
        PlanarFormat.nv12 => 2,
        PlanarFormat.i420 => 3,
        PlanarFormat.yuy2 => 1,
        PlanarFormat.bgra8 => 1,
      };
}

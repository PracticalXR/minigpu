// yuv_helpers.wgsl
//
// Helper functions for YUV color space conversion in WGSL compute shaders.
// Include this in your shader by prepending its contents before your @compute fn.
//
// Supported color matrices:
//   yuv_bt601_full()   — BT.601 full-range (JPEG / webcam common)
//   yuv_bt709_full()   — BT.709 full-range (HD video, default for cameras)
//   yuv_bt2020_full()  — BT.2020 full-range (HDR / WCG)
//
// Usage example (NV12 — two R8Unorm + Rg8Unorm textures):
//   @group(0) @binding(0) var y_plane  : texture_2d<f32>;
//   @group(0) @binding(1) var uv_plane : texture_2d<f32>;
//
//   let rgb = nv12_to_rgb_bt709(y_plane, uv_plane, pixel_coord);

// ─────────────────────────────────────────────────────────────────────────────
// BT.709 full-range coefficients
// ─────────────────────────────────────────────────────────────────────────────
fn nv12_to_rgb_bt709(
    y_tex:  texture_2d<f32>,
    uv_tex: texture_2d<f32>,
    coord:  vec2<i32>
) -> vec3<f32> {
    let y  = textureLoad(y_tex,  coord,              0).r;
    let uv = textureLoad(uv_tex, vec2<i32>(coord.x / 2, coord.y / 2), 0).rg;
    let cb = uv.r - 0.5;
    let cr = uv.g - 0.5;
    let r = y + 1.5748  * cr;
    let g = y - 0.18732 * cb - 0.46812 * cr;
    let b = y + 1.8556  * cb;
    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn nv12_to_rgba_bt709(
    y_tex:  texture_2d<f32>,
    uv_tex: texture_2d<f32>,
    coord:  vec2<i32>
) -> vec4<f32> {
    return vec4<f32>(nv12_to_rgb_bt709(y_tex, uv_tex, coord), 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// BT.601 full-range coefficients
// ─────────────────────────────────────────────────────────────────────────────
fn nv12_to_rgb_bt601(
    y_tex:  texture_2d<f32>,
    uv_tex: texture_2d<f32>,
    coord:  vec2<i32>
) -> vec3<f32> {
    let y  = textureLoad(y_tex,  coord,              0).r;
    let uv = textureLoad(uv_tex, vec2<i32>(coord.x / 2, coord.y / 2), 0).rg;
    let cb = uv.r - 0.5;
    let cr = uv.g - 0.5;
    let r = y + 1.402   * cr;
    let g = y - 0.34414 * cb - 0.71414 * cr;
    let b = y + 1.772   * cb;
    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

// ─────────────────────────────────────────────────────────────────────────────
// I420 / YUV420P three-plane variant (three R8Unorm textures: Y, U, V)
// ─────────────────────────────────────────────────────────────────────────────
fn yuv420p_to_rgb_bt709(
    y_tex: texture_2d<f32>,
    u_tex: texture_2d<f32>,
    v_tex: texture_2d<f32>,
    coord: vec2<i32>
) -> vec3<f32> {
    let y  = textureLoad(y_tex, coord, 0).r;
    let cb = textureLoad(u_tex, vec2<i32>(coord.x / 2, coord.y / 2), 0).r - 0.5;
    let cr = textureLoad(v_tex, vec2<i32>(coord.x / 2, coord.y / 2), 0).r - 0.5;
    let r = y + 1.5748  * cr;
    let g = y - 0.18732 * cb - 0.46812 * cr;
    let b = y + 1.8556  * cb;
    return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

// ─────────────────────────────────────────────────────────────────────────────
// Grayscale — single R8Unorm plane
// ─────────────────────────────────────────────────────────────────────────────
fn gray_to_rgb(y_tex: texture_2d<f32>, coord: vec2<i32>) -> vec3<f32> {
    let y = textureLoad(y_tex, coord, 0).r;
    return vec3<f32>(y, y, y);
}

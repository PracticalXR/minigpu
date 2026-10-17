
struct Dimensions {
  width: u32,
  height: u32,
  padding: vec2<u32>,
}
@group(0) @binding(0) var captured_frame: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> packed_pixels: array<u32>;
@group(0) @binding(2) var<uniform> dimensions: Dimensions;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) pixel: vec3<u32>) {
  if (pixel.x >= dimensions.width || pixel.y >= dimensions.height) { return; }
  let color = textureLoad(captured_frame, vec2<i32>(pixel.xy), 0);
  packed_pixels[pixel.y * dimensions.width + pixel.x] = pack4x8unorm(color);
}

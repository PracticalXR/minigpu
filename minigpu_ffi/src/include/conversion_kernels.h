#pragma once

#include <string>

namespace mgpu {
// Forward declarations to avoid circular dependencies.
class MGPU;
class Buffer;

namespace kernels {

// Kernel to unpack 4x int8 (packed in i32) to 4x int32
const std::string kPackedInt8ToInt32Kernel = R"(
  @group(0) @binding(0) var<storage, read_write> packed_input: array<i32>; // Use read
  @group(0) @binding(1) var<storage, read_write> unpacked_output: array<i32>;

  // Function to sign-extend an 8-bit value (represented in the lower bits of an i32)
  fn sign_extend_i8(val: i32) -> i32 {
    return (val << 24) >> 24;
  }

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let packed_idx: u32 = gid.x;
    let logical_length = arrayLength(&unpacked_output); // Get logical length from output buffer

    // Check bounds for the PACKED input array
    if (packed_idx >= arrayLength(&packed_input)) {
      return;
    }

    let packed_val = packed_input[packed_idx];
    let base_output_idx = packed_idx * 4u;

    // Unpack and write elements individually, checking logical bounds
    let idx0 = base_output_idx + 0u;
    if (idx0 < logical_length) {
      unpacked_output[idx0] = sign_extend_i8((packed_val >> 0u) & 0xFF);
    }

    let idx1 = base_output_idx + 1u;
    if (idx1 < logical_length) {
      unpacked_output[idx1] = sign_extend_i8((packed_val >> 8u) & 0xFF);
    }

    let idx2 = base_output_idx + 2u;
    if (idx2 < logical_length) {
      unpacked_output[idx2] = sign_extend_i8((packed_val >> 16u) & 0xFF);
    }

    let idx3 = base_output_idx + 3u;
    if (idx3 < logical_length) {
      unpacked_output[idx3] = sign_extend_i8((packed_val >> 24u) & 0xFF);
    }
  }
)";

// Kernel to pack 4x int32 (using only lower 8 bits) into 1x int32.
const std::string kInt32ToPackedInt8Kernel = R"(
  @group(0) @binding(0) var<storage, read_write> unpacked_input: array<i32>; // Use read
  @group(0) @binding(1) var<storage, read_write> packed_output: array<i32>;

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let packed_idx: u32 = gid.x; // Index for the PACKED output array
    let logical_length = arrayLength(&unpacked_input); // Get logical length from input buffer

    // Check bounds for the PACKED output array
     if (packed_idx >= arrayLength(&packed_output)) {
      return;
    }

    let base_input_idx = packed_idx * 4u;
    var packed_result: i32 = 0;

    // Read and pack elements individually, checking logical bounds
    let idx0 = base_input_idx + 0u;
    if (idx0 < logical_length) {
      let val0 = unpacked_input[idx0];
      packed_result = packed_result | ((val0 & 0xFF) << 0u);
    }

    let idx1 = base_input_idx + 1u;
    if (idx1 < logical_length) {
      let val1 = unpacked_input[idx1];
      packed_result = packed_result | ((val1 & 0xFF) << 8u);
    }

    let idx2 = base_input_idx + 2u;
    if (idx2 < logical_length) {
      let val2 = unpacked_input[idx2];
      packed_result = packed_result | ((val2 & 0xFF) << 16u);
    }

    let idx3 = base_input_idx + 3u;
    if (idx3 < logical_length) {
      let val3 = unpacked_input[idx3];
      packed_result = packed_result | ((val3 & 0xFF) << 24u);
    }

    packed_output[packed_idx] = packed_result;
  }
)";

const std::string kPackedUInt8ToUInt32Kernel = R"(
    @group(0) @binding(0) var<storage, read_write> packed_input: array<u32>;
    @group(0) @binding(1) var<storage, read_write> unpacked_output: array<u32>;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let packed_idx: u32 = gid.x;
      let logical_length = arrayLength(&unpacked_output);

      if (packed_idx >= arrayLength(&packed_input)) { return; }

      let packed_val = packed_input[packed_idx];
      let base_output_idx = packed_idx * 4u;

      let idx0 = base_output_idx + 0u;
      if (idx0 < logical_length) { unpacked_output[idx0] = (packed_val >> 0u) & 0xFFu; }

      let idx1 = base_output_idx + 1u;
      if (idx1 < logical_length) { unpacked_output[idx1] = (packed_val >> 8u) & 0xFFu; }

      let idx2 = base_output_idx + 2u;
      if (idx2 < logical_length) { unpacked_output[idx2] = (packed_val >> 16u) & 0xFFu; }

      let idx3 = base_output_idx + 3u;
      if (idx3 < logical_length) { unpacked_output[idx3] = (packed_val >> 24u) & 0xFFu; }
    }
  )";

// Pack 4x uint32 (lower 8 bits) into 1x uint32
const std::string kUInt32ToPackedUInt8Kernel = R"(
    @group(0) @binding(0) var<storage, read_write> unpacked_input: array<u32>;
    @group(0) @binding(1) var<storage, read_write> packed_output: array<u32>;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let packed_idx: u32 = gid.x;
      let logical_length = arrayLength(&unpacked_input);

      if (packed_idx >= arrayLength(&packed_output)) { return; }

      let base_input_idx = packed_idx * 4u;
      var packed_result: u32 = 0u;

      let idx0 = base_input_idx + 0u;
      if (idx0 < logical_length) { packed_result = packed_result | ((unpacked_input[idx0] & 0xFFu) << 0u); }

      let idx1 = base_input_idx + 1u;
      if (idx1 < logical_length) { packed_result = packed_result | ((unpacked_input[idx1] & 0xFFu) << 8u); }

      let idx2 = base_input_idx + 2u;
      if (idx2 < logical_length) { packed_result = packed_result | ((unpacked_input[idx2] & 0xFFu) << 16u); }

      let idx3 = base_input_idx + 3u;
      if (idx3 < logical_length) { packed_result = packed_result | ((unpacked_input[idx3] & 0xFFu) << 24u); }

      packed_output[packed_idx] = packed_result;
    }
  )";

// --- 16-bit Kernels ---

// Unpack 2x int16 (packed in i32) to 2x int32 (Sign Extended)
const std::string kPackedInt16ToInt32Kernel = R"(
    @group(0) @binding(0) var<storage, read_write> packed_input: array<i32>;
    @group(0) @binding(1) var<storage, read_write> unpacked_output: array<i32>;

    fn sign_extend_i16(val: i32) -> i32 { return (val << 16) >> 16; }

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let packed_idx: u32 = gid.x;
      let logical_length = arrayLength(&unpacked_output);

      if (packed_idx >= arrayLength(&packed_input)) { return; }

      let packed_val = packed_input[packed_idx];
      let base_output_idx = packed_idx * 2u;

      let idx0 = base_output_idx + 0u;
      if (idx0 < logical_length) { unpacked_output[idx0] = sign_extend_i16((packed_val >> 0u) & 0xFFFF); }

      let idx1 = base_output_idx + 1u;
      if (idx1 < logical_length) { unpacked_output[idx1] = sign_extend_i16((packed_val >> 16u) & 0xFFFF); }
    }
  )";

// Pack 2x int32 (lower 16 bits) into 1x int32
const std::string kInt32ToPackedInt16Kernel = R"(
    @group(0) @binding(0) var<storage, read_write> unpacked_input: array<i32>;
    @group(0) @binding(1) var<storage, read_write> packed_output: array<i32>;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let packed_idx: u32 = gid.x;
      let logical_length = arrayLength(&unpacked_input);

      if (packed_idx >= arrayLength(&packed_output)) { return; }

      let base_input_idx = packed_idx * 2u;
      var packed_result: i32 = 0;

      let idx0 = base_input_idx + 0u;
      if (idx0 < logical_length) { packed_result = packed_result | ((unpacked_input[idx0] & 0xFFFF) << 0u); }

      let idx1 = base_input_idx + 1u;
      if (idx1 < logical_length) { packed_result = packed_result | ((unpacked_input[idx1] & 0xFFFF) << 16u); }

      packed_output[packed_idx] = packed_result;
    }
  )";

// Unpack 2x uint16 (packed in u32) to 2x uint32 (Zero Extended)
const std::string kPackedUInt16ToUInt32Kernel = R"(
  @group(0) @binding(0) var<storage, read_write> packed_input: array<u32>; // Read-only input
  @group(0) @binding(1) var<storage, read_write> unpacked_output: array<u32>;

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let packed_idx: u32 = gid.x;
    let logical_length = arrayLength(&unpacked_output); // Get logical length from output

    // Check bounds for the PACKED input array
    if (packed_idx >= arrayLength(&packed_input)) { return; }

    let packed_val = packed_input[packed_idx];
    let base_output_idx = packed_idx * 2u;

    // Check bounds for EACH output element individually
    let idx0 = base_output_idx + 0u;
    if (idx0 < logical_length) {
      unpacked_output[idx0] = (packed_val >> 0u) & 0xFFFFu;
    }

    let idx1 = base_output_idx + 1u;
    if (idx1 < logical_length) {
      unpacked_output[idx1] = (packed_val >> 16u) & 0xFFFFu;
    }
  }
)";

// Pack 2x uint32 (lower 16 bits) into 1x uint32
const std::string kUInt32ToPackedUInt16Kernel = R"(
  @group(0) @binding(0) var<storage, read_write> unpacked_input: array<u32>; // Read-only input
  @group(0) @binding(1) var<storage, read_write> packed_output: array<u32>;

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let packed_idx: u32 = gid.x;
    let logical_length = arrayLength(&unpacked_input); // Get logical length from input

    // Check bounds for the PACKED output array
    if (packed_idx >= arrayLength(&packed_output)) { return; }

    let base_input_idx = packed_idx * 2u;
    var packed_result: u32 = 0u;

    // Read and pack elements individually, checking logical bounds
    let idx0 = base_input_idx + 0u;
    if (idx0 < logical_length) {
      packed_result = packed_result | ((unpacked_input[idx0] & 0xFFFFu) << 0u);
    }

    let idx1 = base_input_idx + 1u;
    if (idx1 < logical_length) {
      packed_result = packed_result | ((unpacked_input[idx1] & 0xFFFFu) << 16u);
    }

    packed_output[packed_idx] = packed_result;
  }
)";

// --- 64-bit Kernels (using f64/u32 pairs - requires f64 extension or
// simulation) ---

// Expand 1x f64 into 2x u32 (bitcast)
const std::string kExpandFloat64ToUInt32x2Kernel = R"(

    @group(0) @binding(0) var<storage, read_write> input_f64: array<vec2<u32>>; // Read-only input
    @group(0) @binding(1) var<storage, read_write> output_u32x2: array<vec2<u32>>; // Or array<u32> with stride
  
    @compute @workgroup_size(256) // Adjust size based on performance
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let idx_f64: u32 = gid.x;
  
      if (idx_f64 >= arrayLength(&input_f64)) { return; }
  
      let val_f64 = input_f64[idx_f64];
      let val_u32x2 = bitcast<vec2<u32>>(val_f64); // Bitcast f64 to two u32s
  
      // Assuming output is array<vec2<u32>>
      let idx_u32x2 = idx_f64;
       if (idx_u32x2 >= arrayLength(&output_u32x2)) { return; }
      output_u32x2[idx_u32x2] = val_u32x2;
  
      // // Alternative: If output is array<u32>
      // let base_idx_u32 = idx_f64 * 2u;
      // if ((base_idx_u32 + 1u) >= arrayLength(&output_u32x2)) { return; }
      // output_u32x2[base_idx_u32 + 0u] = val_u32x2.x;
      // output_u32x2[base_idx_u32 + 1u] = val_u32x2.y;
    }
  )";

// Combine 2x u32 into 1x f64 (bitcast)
const std::string kCombineUInt32x2ToFloat64Kernel = R"(  
    @group(0) @binding(0) var<storage, read_write> input_u32x2: array<vec2<u32>>; // Or array<u32>
    @group(0) @binding(1) var<storage, read_write> output_f64: array<vec2<u32>>;
  
    @compute @workgroup_size(256) // Adjust size
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let idx_f64: u32 = gid.x; // Index for the output f64 array
  
      if (idx_f64 >= arrayLength(&output_f64)) { return; }
  
      // Assuming input is array<vec2<u32>>
      let idx_u32x2 = idx_f64;
      if (idx_u32x2 >= arrayLength(&input_u32x2)) {
          // Handle incomplete data? Write 0.0?
          output_f64[idx_f64] = 0.0;
          return;
      }
      let val_u32x2 = input_u32x2[idx_u32x2];
  
      // // Alternative: If input is array<u32>
      // let base_idx_u32 = idx_f64 * 2u;
      // if ((base_idx_u32 + 1u) >= arrayLength(&input_u32x2)) {
      //     output_f64[idx_f64] = 0.0; return;
      // }
      // let val_u32x2 = vec2<u32>(input_u32x2[base_idx_u32 + 0u], input_u32x2[base_idx_u32 + 1u]);
  
  
      let val_f64 = bitcast<f64>(val_u32x2); // Bitcast two u32s to f64
  
      output_f64[idx_f64] = val_f64;
    }
  )";

// --- C++ Dispatch Function Declarations ---

// 8-bit
void dispatchPackedI8toI32(::mgpu::MGPU &mgpu, ::mgpu::Buffer &packed_input,
                           ::mgpu::Buffer &unpacked_output);
void dispatchI32toPackedI8(::mgpu::MGPU &mgpu, ::mgpu::Buffer &unpacked_input,
                           ::mgpu::Buffer &packed_output);
void dispatchPackedU8toU32(::mgpu::MGPU &mgpu, ::mgpu::Buffer &packed_input,
                           ::mgpu::Buffer &unpacked_output);
void dispatchU32toPackedU8(::mgpu::MGPU &mgpu, ::mgpu::Buffer &unpacked_input,
                           ::mgpu::Buffer &packed_output);

// 16-bit
void dispatchPackedI16toI32(::mgpu::MGPU &mgpu, ::mgpu::Buffer &packed_input,
                            ::mgpu::Buffer &unpacked_output);
void dispatchI32toPackedI16(::mgpu::MGPU &mgpu, ::mgpu::Buffer &unpacked_input,
                            ::mgpu::Buffer &packed_output);
void dispatchPackedU16toU32(::mgpu::MGPU &mgpu, ::mgpu::Buffer &packed_input,
                            ::mgpu::Buffer &unpacked_output);
void dispatchU32toPackedU16(::mgpu::MGPU &mgpu, ::mgpu::Buffer &unpacked_input,
                            ::mgpu::Buffer &packed_output);

// 64-bit (f64 example)
void dispatchExpandF64toU32x2(::mgpu::MGPU &mgpu, ::mgpu::Buffer &input_f64,
                              ::mgpu::Buffer &output_u32x2);
void dispatchCombineU32x2toF64(::mgpu::MGPU &mgpu, ::mgpu::Buffer &input_u32x2,
                               ::mgpu::Buffer &output_f64);

} // namespace kernels
} // namespace mgpu
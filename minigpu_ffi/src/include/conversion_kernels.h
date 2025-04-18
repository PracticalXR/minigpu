#pragma once

#include <string>

namespace mgpu {
// Forward declarations to avoid circular dependencies.
class MGPU;
class Buffer;

namespace kernels {

// Kernel to unpack 4x int8 (packed in i32) to 4x int32
const std::string kPackedInt8ToInt32Kernel = R"(
  @group(0) @binding(0) var<storage, read_write> packed_input: array<i32>;
  @group(0) @binding(1) var<storage, read_write> unpacked_output: array<i32>;
  
  // Function to sign-extend an 8-bit value (represented in the lower bits of an i32)
  fn sign_extend_i8(val: i32) -> i32 {
    return (val << 24) >> 24;
  }
  
  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let packed_idx: u32 = gid.x;
  
    // Check bounds for the PACKED input array
    if (packed_idx >= arrayLength(&packed_input)) {
      return;
    }
  
    let packed_val = packed_input[packed_idx];
  
    // Unpack and write 4 separate i32 values
    // Ensure the output buffer is large enough (4x the packed size)
    let base_output_idx = packed_idx * 4u;
  
    // Check bounds for the UNPACKED output array (optional but safer)
    // This assumes arrayLength(&unpacked_output) is at least 4 * arrayLength(&packed_input)
    if ((base_output_idx + 3u) >= arrayLength(&unpacked_output)) {
        return; // Avoid out-of-bounds write if something is wrong
    }
  
    unpacked_output[base_output_idx + 0u] = sign_extend_i8((packed_val >> 0u) & 0xFF);
    unpacked_output[base_output_idx + 1u] = sign_extend_i8((packed_val >> 8u) & 0xFF);
    unpacked_output[base_output_idx + 2u] = sign_extend_i8((packed_val >> 16u) & 0xFF);
    unpacked_output[base_output_idx + 3u] = sign_extend_i8((packed_val >> 24u) & 0xFF);
  }
  )";

// Kernel to pack 4x int32 (using only lower 8 bits) into 1x int32.
const std::string kInt32ToPackedInt8Kernel = R"(
  @group(0) @binding(0) var<storage, read_write> unpacked_input: array<i32>;
  @group(0) @binding(1) var<storage, read_write> packed_output: array<i32>;
  
  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let packed_idx: u32 = gid.x; // Index for the PACKED output array
  
    // Check bounds for the PACKED output array
     if (packed_idx >= arrayLength(&packed_output)) {
      return;
    }
  
    let base_input_idx = packed_idx * 4u;
  
    // Check bounds for the UNPACKED input array (optional but safer)
    // Assumes arrayLength(&unpacked_input) is at least 4 * arrayLength(&packed_output)
     if ((base_input_idx + 3u) >= arrayLength(&unpacked_input)) {
        // Handle potential error or incomplete data - maybe write 0?
        packed_output[packed_idx] = 0;
        return;
    }
  
    // Read 4 separate i32 values
    let val0 = unpacked_input[base_input_idx + 0u];
    let val1 = unpacked_input[base_input_idx + 1u];
    let val2 = unpacked_input[base_input_idx + 2u];
    let val3 = unpacked_input[base_input_idx + 3u];
  
    // Pack the lower 8 bits of each into one i32
    var packed_result: i32 = 0;
    packed_result = packed_result | ((val0 & 0xFF) << 0u);
    packed_result = packed_result | ((val1 & 0xFF) << 8u);
    packed_result = packed_result | ((val2 & 0xFF) << 16u);
    packed_result = packed_result | ((val3 & 0xFF) << 24u);
  
    packed_output[packed_idx] = packed_result;
  }
  )";

// Kernel to add 1 to each element of an int32 array.
const std::string kAddOneToInt32Kernel = R"(
  @group(0) @binding(0) var<storage, read_write> data: array<i32>;

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx: u32 = gid.x;

    // Check bounds
    if (idx >= arrayLength(&data)) {
      return;
    }

    data[idx] = data[idx] + 1;
  }
  )";

// Helper function declarations using fully qualified types.
void dispatchPackedI8toI32(::mgpu::MGPU &mgpu, ::mgpu::Buffer &packed,
                           ::mgpu::Buffer &unpacked);

void dispatchI32toPackedI8(::mgpu::MGPU &mgpu, ::mgpu::Buffer &packed,
                           ::mgpu::Buffer &unpacked);

void dispatchAddOneToInt32(::mgpu::MGPU &mgpu, ::mgpu::Buffer &buffer);

} // namespace kernels
} // namespace mgpu
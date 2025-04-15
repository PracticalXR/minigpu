#pragma once

#include <string>

namespace mgpu {
namespace kernels {

// Unpacks 4 x int8 stored in one i32 -> 4 x i32 (sign extended)
const std::string kUnpackI32toI8 = R"(
    @group(0) @binding(0) var<storage, read_write> packed_input: array<i32>;
    @group(0) @binding(1) var<storage, read_write> unpacked_output: array<i32>;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let packed_idx = gid.x;
        let num_packed_elements = arrayLength(&packed_input);

        if (packed_idx >= num_packed_elements) { return; }

        let packed_val = packed_input[packed_idx];
        let base_unpacked_idx = packed_idx * 4u;
        let num_unpacked_elements = arrayLength(&unpacked_output);

        // Byte 0
        if (base_unpacked_idx < num_unpacked_elements) {
            var val = i32(packed_val & 0xFF);
            if ((val & 0x80) != 0) { val = val | 0xFFFFFF00; } // Sign extend
            unpacked_output[base_unpacked_idx] = val;
        }
        // Byte 1
        if (base_unpacked_idx + 1u < num_unpacked_elements) {
            var val = i32((packed_val >> 8) & 0xFF);
             if ((val & 0x80) != 0) { val = val | 0xFFFFFF00; }
            unpacked_output[base_unpacked_idx + 1u] = val;
        }
        // Byte 2
        if (base_unpacked_idx + 2u < num_unpacked_elements) {
            var val = i32((packed_val >> 16) & 0xFF);
             if ((val & 0x80) != 0) { val = val | 0xFFFFFF00; }
            unpacked_output[base_unpacked_idx + 2u] = val;
        }
        // Byte 3
        if (base_unpacked_idx + 3u < num_unpacked_elements) {
            var val = i32((packed_val >> 24) & 0xFF);
             if ((val & 0x80) != 0) { val = val | 0xFFFFFF00; }
            unpacked_output[base_unpacked_idx + 3u] = val;
        }
    }
)";

// Unpacks 4 x uint8 stored in one u32 -> 4 x u32
const std::string kUnpackU32toU8 = R"(
    @group(0) @binding(0) var<storage, read_write> packed_input: array<u32>;
    @group(0) @binding(1) var<storage, read_write> unpacked_output: array<u32>;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let packed_idx = gid.x;
        let num_packed_elements = arrayLength(&packed_input);

        if (packed_idx >= num_packed_elements) { return; }

        let packed_val = packed_input[packed_idx];
        let base_unpacked_idx = packed_idx * 4u;
        let num_unpacked_elements = arrayLength(&unpacked_output);

        if (base_unpacked_idx < num_unpacked_elements) {
            unpacked_output[base_unpacked_idx] = packed_val & 0xFFu;
        }
        if (base_unpacked_idx + 1u < num_unpacked_elements) {
            unpacked_output[base_unpacked_idx + 1u] = (packed_val >> 8u) & 0xFFu;
        }
        if (base_unpacked_idx + 2u < num_unpacked_elements) {
            unpacked_output[base_unpacked_idx + 2u] = (packed_val >> 16u) & 0xFFu;
        }
        if (base_unpacked_idx + 3u < num_unpacked_elements) {
            unpacked_output[base_unpacked_idx + 3u] = (packed_val >> 24u) & 0xFFu;
        }
    }
)";

// Add kUnpackI32toI16, kUnpackU32toU16 similarly...

// Packs 4 x i32 (representing i8) -> 1 x i32
const std::string kPackI8toI32 = R"(
    @group(0) @binding(0) var<storage, read_write> unpacked_input: array<i32>;
    @group(0) @binding(1) var<storage, read_write> packed_output: array<i32>;

    @compute @workgroup_size(256)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let packed_idx = gid.x;
        let num_packed_elements = arrayLength(&packed_output);

        if (packed_idx >= num_packed_elements) { return; }

        let base_unpacked_idx = packed_idx * 4u;
        let num_unpacked_elements = arrayLength(&unpacked_input);

        var packed_val: i32 = 0;

        if (base_unpacked_idx < num_unpacked_elements) {
            packed_val = packed_val | (unpacked_input[base_unpacked_idx] & 0xFF);
        }
        if (base_unpacked_idx + 1u < num_unpacked_elements) {
             packed_val = packed_val | ((unpacked_input[base_unpacked_idx + 1u] & 0xFF) << 8);
        }
         if (base_unpacked_idx + 2u < num_unpacked_elements) {
             packed_val = packed_val | ((unpacked_input[base_unpacked_idx + 2u] & 0xFF) << 16);
        }
         if (base_unpacked_idx + 3u < num_unpacked_elements) {
             packed_val = packed_val | ((unpacked_input[base_unpacked_idx + 3u] & 0xFF) << 24);
        }
        packed_output[packed_idx] = packed_val;
    }
)";
// Add kPackU8toU32, kPackI16toI32, kPackU16toU32 similarly...

} // namespace kernels
} // namespace mgpu
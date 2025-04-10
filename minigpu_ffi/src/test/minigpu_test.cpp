#include "../include/minigpu.h"
#include <iostream>

void testCreateContext() {
  std::cout << "Testing context creation..." << std::endl;
  mgpuInitializeContext();
  // Assuming no error is reported, the context is created.
  std::cout << "Context created successfully." << std::endl;
}

void testCreateBuffer() {
  std::cout << "Testing buffer creation (1024 bytes)..." << std::endl;
  MGPUBuffer *buffer = mgpuCreateBuffer(1024);
  if (buffer) {
    std::cout << "Buffer created successfully." << std::endl;
    mgpuDestroyBuffer(buffer);
    std::cout << "Buffer destroyed successfully." << std::endl;
  } else {
    std::cerr << "Failed to create buffer!" << std::endl;
  }
}

void testComputeShader() {
  std::cout << "Testing compute shader..." << std::endl;
  MGPUComputeShader *shader = mgpuCreateComputeShader();
  if (!shader) {
    std::cerr << "Failed to create compute shader." << std::endl;
    return;
  }

  // Load a basic kernel string.
  const char *kernelCode = R"(
        const GELU_SCALING_FACTOR: f32 = 0.7978845608028654;
        @group(0) @binding(0) var<storage, read_write> inp: array<f32>;
        @group(0) @binding(1) var<storage, read_write> out: array<f32>;
        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
            let i: u32 = GlobalInvocationID.x;
            if (i < 100u) {
                let x: f32 = inp[i];
                out[i] = x + 0.2;
            }
        }
    )";

  mgpuLoadKernel(shader, kernelCode);

  // Create buffers for 100 floats.
  const int numFloats = 100;
  MGPUBuffer *inpBuffer = mgpuCreateBuffer(numFloats * sizeof(float));
  MGPUBuffer *outBuffer = mgpuCreateBuffer(numFloats * sizeof(float));
  if (!inpBuffer || !outBuffer) {
    std::cerr << "Failed to create one or more buffers." << std::endl;
    mgpuDestroyComputeShader(shader);
    return;
  }

  // Initialize input data.
  float inputData[numFloats];
  for (int i = 0; i < numFloats; i++) {
    inputData[i] = static_cast<float>(i);
  }
  // Use the API call to set buffer data.
  mgpuSetBufferDataFloat(inpBuffer, inputData, numFloats * sizeof(float));

  // Set buffers on the shader.
  // Here tag '0' for the input buffer and tag '1' for the output buffer.
  mgpuSetBuffer(shader, 0, inpBuffer);
  mgpuSetBuffer(shader, 1, outBuffer);

  // Dispatch the compute shader (using 1 workgroup for testing).
  mgpuDispatch(shader, 1, 1, 1);
  std::cout << "Compute shader dispatched successfully." << std::endl;

  // Read the output data synchronously.
  float outputData[numFloats] = {0};
  mgpuReadBufferSync(outBuffer, outputData, numFloats * sizeof(float), 0);

  // Print output for verification.
  std::cout << "Buffer input values + 0.2 (expected results):" << std::endl;
  for (int i = 0; i < numFloats; i++) {
    std::cout << "Index " << i << ": " << outputData[i] << std::endl;
  }

  // Clean up resources.
  mgpuDestroyBuffer(inpBuffer);
  mgpuDestroyBuffer(outBuffer);
  mgpuDestroyComputeShader(shader);
}

void testDestroyContext() {
  std::cout << "Testing context destruction..." << std::endl;
  mgpuDestroyContext();
  std::cout << "Context destroyed successfully." << std::endl;
}

void testUint8() {
  std::cout << "Testing uint8 buffer..." << std::endl;
  const int numElements = 10;
  // Create a buffer with 10 bytes.
  MGPUBuffer *buffer = mgpuCreateBuffer(numElements);
  assert(buffer && "Failed to create uint8 buffer");

  // Create input data (uint8_t).
  uint8_t inputData[numElements] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  // Set data for the uint8 buffer.
  mgpuSetBufferDataUint8(buffer, inputData, numElements);

  // Read back data.
  uint8_t outputData[numElements] = {0};
  mgpuReadBufferSync(buffer, outputData, numElements, 0);

  // Validate that the output data matches the input.
  if (memcmp(inputData, outputData, numElements) != 0) {
    std::cerr << "Uint8 test failed: output does not match input." << std::endl;
    exit(1);
  }
  std::cout << "Uint8 test passed." << std::endl;
  mgpuDestroyBuffer(buffer);
}

void testInt8() {
  std::cout << "Testing int8 buffer..." << std::endl;
  const int numElements = 10;
  // Create a buffer with 10 bytes.
  MGPUBuffer *buffer = mgpuCreateBuffer(numElements);
  assert(buffer && "Failed to create int8 buffer");

  // Create input data (int8_t).
  int8_t inputData[numElements] = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10};

  // Set data for the int8 buffer.
  mgpuSetBufferDataInt8(buffer, inputData, numElements);

  // Read back data.
  int8_t outputData[numElements] = {0};
  mgpuReadBufferSync(buffer, outputData, numElements, 0);

  // Validate that the output data matches the input.
  if (memcmp(inputData, outputData, numElements) != 0) {
    std::cerr << "Int8 test failed: output does not match input." << std::endl;
    exit(1);
  }
  std::cout << "Int8 test passed." << std::endl;
  mgpuDestroyBuffer(buffer);
}

int main() {
  // Initialize GPU context.
  mgpuInitializeContext();

  testUint8();
  testInt8();

  // Destroy GPU context.
  mgpuDestroyContext();

  return 0;
}

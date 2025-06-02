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
  MGPUBuffer *buffer = mgpuCreateBuffer(1024, gpu::kf32);
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
  MGPUBuffer *inpBuffer =
      mgpuCreateBuffer(numFloats * sizeof(float), gpu::kf32);
  MGPUBuffer *outBuffer =
      mgpuCreateBuffer(numFloats * sizeof(float), gpu::kf32);
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
  MGPUBuffer *buffer = mgpuCreateBuffer(numElements, gpu::ku8);
  assert(buffer && "Failed to create uint8 buffer");

  // Create input data (uint8_t).
  uint8_t inputData[numElements] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  // Set data for the uint8 buffer.
  mgpuSetBufferDataUint8(buffer, inputData, numElements);

  // Read back data.
  uint8_t outputData[numElements] = {0};
  mgpuReadBufferSyncUint8(buffer, outputData, numElements, 0);

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
  MGPUBuffer *buffer = mgpuCreateBuffer(numElements, gpu::ki8);
  assert(buffer && "Failed to create int8 buffer");

  // Create input data (int8_t).
  int8_t inputData[numElements] = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10};

  // Set data for the int8 buffer.
  mgpuSetBufferDataInt8(buffer, inputData, numElements);

  // Read back data.
  int8_t outputData[numElements] = {0};
  mgpuReadBufferSyncInt8(buffer, outputData, numElements, 0);
}

void testInt16() {
  std::cout << "Testing int16 buffer..." << std::endl;
  const int numElements = 10;
  MGPUBuffer *buffer = mgpuCreateBuffer(numElements * sizeof(int16_t),
                                        3); // ki8 = 3, but we need ki16 = 4
  assert(buffer && "Failed to create int16 buffer");

  int16_t inputData[numElements] = {-100, -200, -300, -400, -500,
                                    600,  700,  800,  900,  1000};
  mgpuSetBufferDataInt16(buffer, inputData, numElements * sizeof(int16_t));

  int16_t outputData[numElements] = {0};
  mgpuReadBufferSyncInt16(buffer, outputData, numElements, 0);

  if (memcmp(inputData, outputData, numElements * sizeof(int16_t)) != 0) {
    std::cerr << "Int16 test failed: output does not match input." << std::endl;
    exit(1);
  }
  std::cout << "Int16 test passed." << std::endl;
  mgpuDestroyBuffer(buffer);
}

void testUint16() {
  std::cout << "Testing uint16 buffer..." << std::endl;
  const int numElements = 10;
  MGPUBuffer *buffer =
      mgpuCreateBuffer(numElements * sizeof(uint16_t), 8); // ku16 = 8
  assert(buffer && "Failed to create uint16 buffer");

  uint16_t inputData[numElements] = {100, 200, 300, 400, 500,
                                     600, 700, 800, 900, 1000};
  mgpuSetBufferDataUint16(buffer, inputData, numElements * sizeof(uint16_t));

  uint16_t outputData[numElements] = {0};
  mgpuReadBufferSyncUint16(buffer, outputData, numElements, 0);

  if (memcmp(inputData, outputData, numElements * sizeof(uint16_t)) != 0) {
    std::cerr << "Uint16 test failed: output does not match input."
              << std::endl;
    exit(1);
  }
  std::cout << "Uint16 test passed." << std::endl;
  mgpuDestroyBuffer(buffer);
}

void testInt32() {
  std::cout << "Testing int32 buffer..." << std::endl;
  const int numElements = 10;
  MGPUBuffer *buffer =
      mgpuCreateBuffer(numElements * sizeof(int32_t), 5); // ki32 = 5
  assert(buffer && "Failed to create int32 buffer");

  int32_t inputData[numElements] = {-1000, -2000, -3000, -4000, -5000,
                                    6000,  7000,  8000,  9000,  10000};
  mgpuSetBufferDataInt32(buffer, inputData, numElements * sizeof(int32_t));

  int32_t outputData[numElements] = {0};
  mgpuReadBufferSyncInt32(buffer, outputData, numElements, 0);

  if (memcmp(inputData, outputData, numElements * sizeof(int32_t)) != 0) {
    std::cerr << "Int32 test failed: output does not match input." << std::endl;
    exit(1);
  }
  std::cout << "Int32 test passed." << std::endl;
  mgpuDestroyBuffer(buffer);
}

void testUint32() {
  std::cout << "Testing uint32 buffer..." << std::endl;
  const int numElements = 10;
  MGPUBuffer *buffer =
      mgpuCreateBuffer(numElements * sizeof(uint32_t), 9); // ku32 = 9
  assert(buffer && "Failed to create uint32 buffer");

  uint32_t inputData[numElements] = {1000, 2000, 3000, 4000, 5000,
                                     6000, 7000, 8000, 9000, 10000};
  mgpuSetBufferDataUint32(buffer, inputData, numElements * sizeof(uint32_t));

  uint32_t outputData[numElements] = {0};
  mgpuReadBufferSyncUint32(buffer, outputData, numElements, 0);

  if (memcmp(inputData, outputData, numElements * sizeof(uint32_t)) != 0) {
    std::cerr << "Uint32 test failed: output does not match input."
              << std::endl;
    exit(1);
  }
  std::cout << "Uint32 test passed." << std::endl;
  mgpuDestroyBuffer(buffer);
}

void testInt64() {
  std::cout << "Testing int64 buffer..." << std::endl;
  const int numElements = 10;
  MGPUBuffer *buffer =
      mgpuCreateBuffer(numElements * sizeof(int64_t), 6); // ki64 = 6
  assert(buffer && "Failed to create int64 buffer");

  int64_t inputData[numElements] = {-100000, -200000, -300000, -400000,
                                    -500000, 600000,  700000,  800000,
                                    900000,  1000000};
  mgpuSetBufferDataInt64(buffer, inputData, numElements * sizeof(int64_t));

  int64_t outputData[numElements] = {0};
  mgpuReadBufferSyncInt64(buffer, outputData, numElements, 0);

  if (memcmp(inputData, outputData, numElements * sizeof(int64_t)) != 0) {
    std::cerr << "Int64 test failed: output does not match input." << std::endl;
    exit(1);
  }
  std::cout << "Int64 test passed." << std::endl;
  mgpuDestroyBuffer(buffer);
}

void testUint64() {
  std::cout << "Testing uint64 buffer..." << std::endl;
  const int numElements = 10;
  MGPUBuffer *buffer =
      mgpuCreateBuffer(numElements * sizeof(uint64_t), 10); // ku64 = 10
  assert(buffer && "Failed to create uint64 buffer");

  uint64_t inputData[numElements] = {100000, 200000, 300000, 400000, 500000,
                                     600000, 700000, 800000, 900000, 1000000};
  mgpuSetBufferDataUint64(buffer, inputData, numElements * sizeof(uint64_t));

  uint64_t outputData[numElements] = {0};
  mgpuReadBufferSyncUint64(buffer, outputData, numElements, 0);

  if (memcmp(inputData, outputData, numElements * sizeof(uint64_t)) != 0) {
    std::cerr << "Uint64 test failed: output does not match input."
              << std::endl;
    exit(1);
  }
  std::cout << "Uint64 test passed." << std::endl;
  mgpuDestroyBuffer(buffer);
}

void testFloat32() {
  std::cout << "Testing float32 buffer..." << std::endl;
  const int numElements = 10;
  MGPUBuffer *buffer =
      mgpuCreateBuffer(numElements * sizeof(float), 1); // kf32 = 1
  assert(buffer && "Failed to create float32 buffer");

  float inputData[numElements] = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f,
                                  6.6f, 7.7f, 8.8f, 9.9f, 10.0f};
  mgpuSetBufferDataFloat(buffer, inputData, numElements * sizeof(float));

  float outputData[numElements] = {0};
  mgpuReadBufferSyncFloat32(buffer, outputData, numElements, 0);

  if (memcmp(inputData, outputData, numElements * sizeof(float)) != 0) {
    std::cerr << "Float32 test failed: output does not match input."
              << std::endl;
    exit(1);
  }
  std::cout << "Float32 test passed." << std::endl;
  mgpuDestroyBuffer(buffer);
}

void testFloat64() {
  std::cout << "Testing float64 buffer..." << std::endl;
  const int numElements = 10;
  MGPUBuffer *buffer =
      mgpuCreateBuffer(numElements * sizeof(double), 2); // kf64 = 2
  assert(buffer && "Failed to create float64 buffer");

  double inputData[numElements] = {1.1, 2.2, 3.3, 4.4, 5.5,
                                   6.6, 7.7, 8.8, 9.9, 10.0};
  mgpuSetBufferDataDouble(buffer, inputData, numElements * sizeof(double));

  double outputData[numElements] = {0};
  mgpuReadBufferSyncFloat64(buffer, outputData, numElements, 0);

  if (memcmp(inputData, outputData, numElements * sizeof(double)) != 0) {
    std::cerr << "Float64 test failed: output does not match input."
              << std::endl;
    exit(1);
  }
  std::cout << "Float64 test passed." << std::endl;
  mgpuDestroyBuffer(buffer);
}

int main() {
  mgpuInitializeContext();

  testUint8();
  testInt8();
  testInt16();
  testUint16();
  testInt32();
  testUint32();
  testInt64();
  testUint64();
  testFloat32();
  testFloat64();
  testCreateBuffer();
  testComputeShader();

  mgpuDestroyContext();
  return 0;
}

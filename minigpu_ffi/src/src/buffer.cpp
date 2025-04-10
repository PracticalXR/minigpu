

#include "../include/buffer.h"
#include "../include/compute_shader.h"
#include "../include/gpuh.h"

using namespace gpu;

namespace mgpu {
void MGPU::initializeContext() {
  try {
    // Wrap context in a unique_ptr.
    ctx = std::make_unique<gpu::Context>(std::move(gpu::createContext()));
    LOG(kDefLog, kInfo, "GPU context initialized successfully.");
  } catch (const std::exception &ex) {
    LOG(kDefLog, kError, "Failed to create GPU context: %s", ex.what());
  }
}

void MGPU::initializeContextAsync(std::function<void()> callback) {
  try {
    initializeContext();
    if (callback) {
      callback();
    }
  } catch (const std::exception &ex) {
    LOG(kDefLog, kError, "Failed to create GPU context: %s", ex.what());
  }
}

void MGPU::destroyContext() {
  if (ctx) {
    ctx.release();
    LOG(kDefLog, kInfo, "GPU context destroyed successfully.");
  } else {
    LOG(kDefLog, kError,
        "GPU context is already destroyed or not initialized.");
  }
}

Buffer::Buffer(MGPU &mgpu) : mgpu(mgpu) {
  bufferData.buffer = nullptr;
  bufferData.usage = 0;
  bufferData.size = 0;
}
void Buffer::createBuffer(int bufferSize) {
  size_t paddedSize = ((bufferSize + 3) / 4) * 4;
  LOG(kDefLog, kInfo, "Creating buffer of size: %d bytes", paddedSize);
  WGPUBufferUsage usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst |
                          WGPUBufferUsage_CopySrc;
  WGPUBufferDescriptor descriptor = {};
  descriptor.usage = usage;
  descriptor.size = paddedSize;
  descriptor.mappedAtCreation = false;
  descriptor.label = {.data = nullptr, .length = 0};

  WGPUBuffer buffer =
      wgpuDeviceCreateBuffer(this->mgpu.getContext().device, &descriptor);
  if (buffer == nullptr) {
    LOG(kDefLog, kError, "Failed to create buffer");
    return;
  }

  bufferData = gpu::Array{
      .buffer = buffer,
      .usage = usage,
      .size = static_cast<size_t>(bufferSize),
  };
}

void Buffer::readSync(void *outputData, size_t size, size_t offset) {
  size_t paddedSize = ((size + 3) / 4) * 4;
  LOG(kDefLog, kInfo, "readSync (float*): Reading %zu bytes from buffer", size);

  gpu::toCPU(this->mgpu.getContext(), bufferData.buffer, outputData, paddedSize,
             offset);

  size_t numFloats = size / sizeof(float);
  if (numFloats > 0) {
    std::string floatString = "readSync (float*): Floats: ";
    for (size_t i = 0; i < numFloats; i++) {
      floatString += std::to_string(static_cast<float *>(outputData)[i]);
      if (i < numFloats - 1)
        floatString += ", ";
    }
    LOG(kDefLog, kInfo, floatString.c_str());
  } else {
    LOG(kDefLog, kInfo,
        "readSync (float*): Not enough data to display float values");
  }
}

void Buffer::readAsync(void *outputData, size_t size, size_t offset,
                       std::function<void()> callback) {
  std::thread([=]() {
    readSync(outputData, size, offset);
    if (callback)
      callback();
  }).detach();
}

void Buffer::setData(const float *inputData, size_t byteSize) {
  size_t paddedSize = ((byteSize + 3) / 4) * 4;
  if (bufferData.buffer == nullptr || byteSize > bufferData.size) {
    createBuffer(paddedSize);
  }
  gpu::toGPU(this->mgpu.getContext(), inputData, bufferData.buffer, paddedSize);
}

void Buffer::setData(const double *inputData, size_t byteSize) {
  if (bufferData.buffer == nullptr || byteSize > bufferData.size) {
    createBuffer(byteSize);
  }
  std::string bufferString = "mgpuSetBufferData (double): ";
  size_t count = byteSize / sizeof(double);
  for (size_t i = 0; i < count; i++) {
    bufferString += std::to_string(inputData[i]);
    if (i < count - 1)
      bufferString += ", ";
  }
  LOG(kDefLog, kInfo, bufferString.c_str());
  gpu::toGPU(this->mgpu.getContext(), inputData, bufferData.buffer, byteSize);
}

void Buffer::setData(const uint8_t *inputData, size_t byteSize) {
  size_t paddedSize = ((byteSize + 3) / 4) * 4;
  if (bufferData.buffer == nullptr || byteSize > bufferData.size) {
    createBuffer(byteSize);
  }
  std::string bufferString = "mgpuSetBufferData (uint8_t): ";
  size_t count = byteSize;
  for (size_t i = 0; i < count; i++) {
    bufferString += std::to_string(inputData[i]);
    if (i < count - 1)
      bufferString += ", ";
  }
  LOG(kDefLog, kInfo, bufferString.c_str());
  LOG(kDefLog, kInfo, "Buffer size: %zu bytes", paddedSize);
  gpu::toGPU(this->mgpu.getContext(), inputData, bufferData.buffer, paddedSize);
}

void Buffer::setData(const uint16_t *inputData, size_t byteSize) {
  if (bufferData.buffer == nullptr || byteSize > bufferData.size) {
    createBuffer(byteSize);
  }
  std::string bufferString = "mgpuSetBufferData (uint16_t): ";
  size_t count = byteSize / sizeof(uint16_t);
  for (size_t i = 0; i < count; i++) {
    bufferString += std::to_string(inputData[i]);
    if (i < count - 1)
      bufferString += ", ";
  }
  LOG(kDefLog, kInfo, bufferString.c_str());
  gpu::toGPU(this->mgpu.getContext(), inputData, bufferData.buffer, byteSize);
}

void Buffer::setData(const uint32_t *inputData, size_t byteSize) {
  if (bufferData.buffer == nullptr || byteSize > bufferData.size) {
    createBuffer(byteSize);
  }
  std::string bufferString = "mgpuSetBufferData (uint32_t): ";
  size_t count = byteSize / sizeof(uint32_t);
  for (size_t i = 0; i < count; i++) {
    bufferString += std::to_string(inputData[i]);
    if (i < count - 1)
      bufferString += ", ";
  }
  LOG(kDefLog, kInfo, bufferString.c_str());
  gpu::toGPU(this->mgpu.getContext(), inputData, bufferData.buffer, byteSize);
}

void Buffer::setData(const uint64_t *inputData, size_t byteSize) {
  if (bufferData.buffer == nullptr || byteSize > bufferData.size) {
    createBuffer(byteSize);
  }
  std::string bufferString = "mgpuSetBufferData (uint64_t): ";
  size_t count = byteSize / sizeof(uint64_t);
  for (size_t i = 0; i < count; i++) {
    bufferString += std::to_string(inputData[i]);
    if (i < count - 1)
      bufferString += ", ";
  }
  LOG(kDefLog, kInfo, bufferString.c_str());
  gpu::toGPU(this->mgpu.getContext(), inputData, bufferData.buffer, byteSize);
}

void Buffer::setData(const int8_t *inputData, size_t byteSize) {
  size_t paddedSize = ((byteSize + 3) / 4) * 4;

  if (bufferData.buffer == nullptr || byteSize > bufferData.size) {
    createBuffer(byteSize);
  }
  std::string bufferString = "mgpuSetBufferData (int8_t): ";
  size_t count = byteSize / sizeof(int8_t);
  for (size_t i = 0; i < count; i++) {
    bufferString += std::to_string(inputData[i]);
    if (i < count - 1)
      bufferString += ", ";
  }
  LOG(kDefLog, kInfo, bufferString.c_str());
  gpu::toGPU(this->mgpu.getContext(), inputData, bufferData.buffer, paddedSize);
}

void Buffer::setData(const int16_t *inputData, size_t byteSize) {
  if (bufferData.buffer == nullptr || byteSize > bufferData.size) {
    createBuffer(byteSize);
  }
  std::string bufferString = "mgpuSetBufferData (int16_t): ";
  size_t count = byteSize / sizeof(int16_t);
  for (size_t i = 0; i < count; i++) {
    bufferString += std::to_string(inputData[i]);
    if (i < count - 1)
      bufferString += ", ";
  }
  LOG(kDefLog, kInfo, bufferString.c_str());
  gpu::toGPU(this->mgpu.getContext(), inputData, bufferData.buffer, byteSize);
}

void Buffer::setData(const int32_t *inputData, size_t byteSize) {
  if (bufferData.buffer == nullptr || byteSize > bufferData.size) {
    createBuffer(byteSize);
  }
  std::string bufferString = "mgpuSetBufferData (int32_t): ";
  size_t count = byteSize / sizeof(int32_t);
  for (size_t i = 0; i < count; i++) {
    bufferString += std::to_string(inputData[i]);
    if (i < count - 1)
      bufferString += ", ";
  }
  LOG(kDefLog, kInfo, bufferString.c_str());
  gpu::toGPU(this->mgpu.getContext(), inputData, bufferData.buffer, byteSize);
}

void Buffer::setData(const int64_t *inputData, size_t byteSize) {
  if (bufferData.buffer == nullptr || byteSize > bufferData.size) {
    createBuffer(byteSize);
  }
  std::string bufferString = "mgpuSetBufferData (int64_t): ";
  size_t count = byteSize / sizeof(int64_t);
  for (size_t i = 0; i < count; i++) {
    bufferString += std::to_string(inputData[i]);
    if (i < count - 1)
      bufferString += ", ";
  }
  LOG(kDefLog, kInfo, bufferString.c_str());
  gpu::toGPU(this->mgpu.getContext(), inputData, bufferData.buffer, byteSize);
}

void Buffer::release() { wgpuBufferRelease(bufferData.buffer); }

} // namespace mgpu
#ifndef COMPUTE_SHADER_H
#define COMPUTE_SHADER_H

#include "buffer.h"

namespace mgpu {
class ComputeShader {
public:
  ComputeShader(MGPU &mgpu);
  void loadKernelString(const std::string &kernelString);
  void loadKernelFile(const std::string &path);
  bool hasKernel() const;
  void setBuffer(int tag, const Buffer &buffer);
  void dispatch(int groupsX, int groupsY, int groupsZ);
  void dispatchAsync(int groupsX, int groupsY, int groupsZ,
                     std::function<void()> callback);
  bool validateAllBuffers() const;
  bool validateSingleBuffer(size_t bindingIndex) const;
  bool validateBufferRelationships() const;
  bool validateMemoryBounds(int groupsX, int groupsY, int groupsZ) const;
  bool validateKernelSpecificConstraints(int groupsX, int groupsY,
                                         int groupsZ) const;
  size_t calculateTotalElements(const gpu::Shape &shape) const;
  ~ComputeShader();

private:
  MGPU &mgpu;
  gpu::KernelCode code;
  std::vector<gpu::Tensor> bindings;
  std::string lastShaderCode;

  std::optional<gpu::Kernel> cachedKernel;
  std::vector<size_t> lastGroupSize = {0, 0, 0};
  size_t lastBindingsHash = 0;

  size_t calculateBindingsHash() const;
  bool needsKernelRecreation(int groupsX, int groupsY, int groupsZ) const;
  void destroyCachedKernel();
};
} // namespace mgpu

#endif // COMPUTE_SHADER_H

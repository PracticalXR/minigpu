#ifndef COMPUTE_SHADER_H
#define COMPUTE_SHADER_H

#include "buffer.h"

namespace mgpu
{
    class ComputeShader
    {
    public:
        ComputeShader(MGPU &mgpu);
        void loadKernelString(const std::string &kernelString);
        void loadKernelFile(const std::string &path);
        bool hasKernel() const;
        void setBuffer(int tag, const Buffer &buffer);
        void dispatch(int groupsX, int groupsY, int groupsZ);
        void dispatchAsync(int groupsX, int groupsY, int groupsZ,
                          std::function<void()> callback);

    private:
        gpu::KernelCode code;
        std::vector<gpu::Tensor> bindings;
        MGPU &mgpu;
    };
}

#endif // COMPUTE_SHADER_H

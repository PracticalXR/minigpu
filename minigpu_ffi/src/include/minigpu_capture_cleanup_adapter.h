#ifndef MINIGPU_CAPTURE_CLEANUP_ADAPTER_H
#define MINIGPU_CAPTURE_CLEANUP_ADAPTER_H
#include "minigpu_capture_cleanup_owner.h"
#include "minigpu_imported_frame.h"
#include "minigpu_private_output.h"
namespace mgpu::capture_cleanup {
// Original lease only. Private C++ routing; no new public/guest authority.
int prepareFamily(MGPUContextLease *, Family &) noexcept;
int destroyShader(MGPUContextLease *, MGPUComputeShader *, MGPUResourceCleanupCallback,
                  void *, const Family &);
int destroyBuffer(MGPUContextLease *, MGPUBuffer *, MGPUResourceCleanupCallback,
                  void *, const Family &);
int prepareUseFamily(MGPUComputeShaderUse *, Family &) noexcept;
int prepareFrameFamily(MGPUImportedFrame *, Family &) noexcept;
void retireUse(MGPUComputeShaderUse *, MGPUQueueCompletionCallback, void *, const Family &);
void retireFrame(MGPUImportedFrame *, MGPUQueueCompletionCallback, void *, const Family &);
int prepareWriteFamily(MGPUPrivateOutputWriter *, MGPUPrivateOutput *, Family &) noexcept;
int prepareWriterFamily(MGPUPrivateOutputWriter *, Family &) noexcept;
int prepareOutputFamily(MGPUPrivateOutput *, Family &) noexcept;
void writeOutput(MGPUPrivateOutputWriter *, MGPUPrivateOutput *, MGPUQueueCompletionCallback,
                 void *, const Family &);
void retireWriter(MGPUPrivateOutputWriter *, MGPUQueueCompletionCallback, void *, const Family &);
void retireOutput(MGPUPrivateOutput *, MGPUQueueCompletionCallback, void *, const Family &);
} // namespace mgpu::capture_cleanup
#endif

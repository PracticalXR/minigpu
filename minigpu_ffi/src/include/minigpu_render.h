#ifndef MINIGPU_RENDER_H
#define MINIGPU_RENDER_H

#include "export.h"
#include "stdint.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MGPUContextHandle MGPUContextHandle;
typedef struct MGPUBuffer MGPUBuffer;
typedef struct MGPUTextureView MGPUTextureView;
typedef struct MGPUSampler MGPUSampler;
typedef struct MGPURenderPipeline MGPURenderPipeline;

EXPORT uint32_t mgpuRasterResourceVersion(void);
EXPORT MGPURenderPipeline *mgpuCreateRenderPipeline(const char *wgsl,
                                                    const char *vertexEntry,
                                                    const char *fragmentEntry,
                                                    const uint32_t *descriptor,
                                                    int wordCount);
EXPORT MGPURenderPipeline *mgpuContextCreateRenderPipeline(
    MGPUContextHandle *context, const char *wgsl, const char *vertexEntry,
    const char *fragmentEntry, const uint32_t *descriptor, int wordCount);
EXPORT int mgpuRenderPipelineStatus(MGPURenderPipeline *pipeline);
// Posts `(token << 1) | ok` to [port] (wire format in minigpu.h) exactly once,
// when [pipeline] leaves status 0: ok when it became ready (1), not ok when
// it failed (2) or was closed (3: destroyed, or its context torn down). A
// pipeline that has already settled posts at once; NULL posts not ok. Read
// mgpuRenderPipelineStatus after the post: a device replaced later still
// turns it to 3. The web build posts nothing (no Dart port); instead every
// settle there calls Module.mgpuRenderPipelineSettled(id), when set, with
// the pipeline handle's value as id (it may fire inside create or destroy).
EXPORT void mgpuRenderPipelineNotifySettledToPort(MGPURenderPipeline *pipeline,
                                                  int64_t port, int64_t token);
EXPORT int mgpuRenderPipelineDiagnostic(MGPURenderPipeline *pipeline,
                                        char *destination, int capacity);
// Monotonic retained-bind-group cache counters: 0 created, 1 reused,
// 2 retired, 3 live. Unknown selectors and null pipelines return zero.
// Double is deliberate: native Dart and Emscripten both receive the same
// JS-number-compatible ABI while retaining exact integer counts through 2^53.
EXPORT double mgpuRenderPipelineBindingStat(MGPURenderPipeline *pipeline,
                                            int selector);
EXPORT void mgpuDestroyRenderPipeline(MGPURenderPipeline *pipeline);
EXPORT int mgpuSubmitRender(MGPURenderPipeline *pipeline,
                            MGPUTextureView *color, MGPUTextureView *depth,
                            const float *clearRgbaDepth, int loadFlags,
                            MGPUBuffer *const *buffers, int bufferCount,
                            MGPUTextureView *const *views, int viewCount,
                            MGPUSampler *const *samplers, int samplerCount,
                            const uint32_t *commands, int wordCount);

#ifdef __cplusplus
}
#endif

#endif // MINIGPU_RENDER_H

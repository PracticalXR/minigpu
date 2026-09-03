#pragma once

// Persistent shader cache — internal interface.
//
// The public C API (mgpuShaderCache*) lives in minigpu.h. This header is the
// C++-only surface that buffer.cpp needs in order to chain a
// WGPUDawnCacheDeviceDescriptor onto the device descriptor, plus the
// instrumentation hook compute_shader.cpp calls around pipeline creation.
//
// WHY THIS EXISTS: Dawn already caches compiled shader blobs (on D3D11 the
// FXC output) through BlobCache, but only if the embedder supplies
// load/store callbacks. Without them every process launch recompiles every
// pipeline from WGSL. Large compute kernels can spend tens of seconds there.
//
// The storage module knows nothing about WebGPU: its only Dawn-facing pieces
// are the two trampolines below, whose signatures are deliberately identical
// to WGPUDawnLoadCacheDataFunction / WGPUDawnStoreCacheDataFunction so
// buffer.cpp can assign them directly without a cast.

#include <cstddef>
#include <cstdint>
#include <string>

namespace mgpu {

/// True when the cache should be wired into the next device creation: caching
/// is enabled AND either a bring-your-own provider is registered or the
/// default disk directory resolved. False means "chain nothing" — device
/// creation then behaves exactly as it did before this feature existed.
bool shaderCacheIsActive();

/// The value handed to DawnCacheDeviceDescriptor::isolationKey, which Dawn
/// folds into every cache key it computes.
///
/// ADDITIVE, never a substitute for Dawn's own key (which already covers Dawn
/// version, backend and toggles). Including the adapter identity here is what
/// makes a GPU swap or driver update a guaranteed MISS at our layer rather
/// than a stale hit, and it lets two contexts on two different adapters share
/// one cache directory safely.
std::string shaderCacheIsolationKey(uint32_t vendorId, uint32_t deviceId,
                                    const std::string &adapterDesc);

/// One INFO line summarising the cache after a device comes up. Per-entry
/// detail is DEBUG only — this log channel defaults to INFO and a line per
/// blob would bury real warnings.
std::string shaderCacheSummaryLine();

/// Accumulates into MGPUShaderCacheStats::pipelineCreateMs. Called around the
/// wgpuDeviceCreateComputePipeline wrapper: with a cold cache this is the
/// compile cost, with a warm one it should collapse. A future regression in
/// kernel compile time shows up here first.
void shaderCacheNotePipelineMs(double ms);

} // namespace mgpu

extern "C" {

/// Dawn's load callback. Two-phase, exactly as BlobCache::LoadInternal drives
/// it: first with (value=nullptr, valueSize=0) to size the entry — return 0
/// for a miss — then with a buffer of that size. Returning a DIFFERENT size on
/// the second call is treated as a miss by Dawn, which is the designed escape
/// hatch for an entry that changed or vanished in between; never block or
/// retry to avoid it.
size_t mgpu_shader_cache_dawn_load(const void *key, size_t keySize, void *value,
                                   size_t valueSize, void *userdata);

/// Dawn's store callback. Best effort by contract: a failure here must cost a
/// recompile next launch and nothing else.
void mgpu_shader_cache_dawn_store(const void *key, size_t keySize,
                                  const void *value, size_t valueSize,
                                  void *userdata);

} // extern "C"

// Cold real-library ABI boundaries only: no synthetic authority, GPU or capture.
#include "../include/minigpu.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
extern "C" int imported_frame_c_abi(void);

static void require(bool value, const char *message) {
  if (!value) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); }
}
static int callbacks = 0;
static void refusedRetirement(int result, void *) {
  require(result == MGPU_CONTEXT_LEASE_INVALID, "missing frame never retires");
  MGPUImportedFrameLifetimeStats stats{};
  require(mgpuImportedFrameGetLifetimeStats(&stats, sizeof(stats)) == 0,
          "refusal callback may reenter diagnostic API");
  ++callbacks;
}
int main() {
  static_assert(sizeof(MGPUImportedFrameInfo) == 48);
  static_assert(sizeof(MGPUImportedFrameLifetimeStats) == 40);
  MGPUContextLifetimeStats beforeContext{}, afterContext{};
  MGPUImportedFrameLifetimeStats before{}, after{};
  require(mgpuContextGetLifetimeStats(&beforeContext, sizeof(beforeContext)) == 0 &&
          mgpuImportedFrameGetLifetimeStats(&before, sizeof(before)) == 0,
          "read cold native ownership accounting");
  const auto refs = mgpuContextRefCount();
  require(before.version == 1 && before.max_frames == 256 &&
          before.max_frame_bytes == 128ull * 1024 * 1024 &&
          before.max_bytes == 512ull * 1024 * 1024, "fixed native retention bounds");
  require(imported_frame_c_abi() == 1, "C translation unit layout and actual ABI");
  auto refuse = [](const MGPUExternalVideoBuffer *d, uint32_t size) {
    auto *out = reinterpret_cast<MGPUImportedFrame *>(uintptr_t(1));
    require(mgpuContextLeaseImportImmutableD3D11BgraFrame(nullptr, d, size, &out) ==
                MGPU_CONTEXT_LEASE_INVALID && !out,
            "cold descriptor/owner boundary refuses and clears output");
  };
  refuse(nullptr, 0);
  MGPUExternalVideoBuffer descriptor{};
  refuse(&descriptor, sizeof(descriptor) - 1);
  refuse(&descriptor, sizeof(descriptor));
  descriptor.content_type = MGPU_EXTERNAL_CONTENT_TYPE_D3D11_SHARED_HANDLE;
  descriptor.pixel_format = MGPU_EXTERNAL_PIXEL_FORMAT_BGRA32;
  descriptor.width = descriptor.height = 1;
  descriptor.num_planes = 1;
  descriptor.planes[0].data_ptr = reinterpret_cast<void *>(uintptr_t(1));
  descriptor.planes[0].width = descriptor.planes[0].height = 1;
  descriptor.planes[0].dmabuf_fd = -1;
  descriptor.planes[0].drm_format_modifier = MGPU_D3D11_IMMUTABLE_READY_TAG;
#ifdef _WIN32
  refuse(&descriptor, sizeof(descriptor)); // No original lease; handle not opened.
#else
  MGPUImportedFrame *unsupported = nullptr;
  require(mgpuContextLeaseImportImmutableD3D11BgraFrame(nullptr, &descriptor,
      sizeof(descriptor), &unsupported) == MGPU_CONTEXT_LEASE_UNSUPPORTED &&
      !unsupported, "unsupported backend creates no owner");
#endif
  descriptor.width = descriptor.planes[0].width = std::numeric_limits<uint32_t>::max();
  descriptor.height = descriptor.planes[0].height = std::numeric_limits<uint32_t>::max();
  refuse(&descriptor, sizeof(descriptor));
  require(mgpuComputeShaderUseBindImportedFrame(nullptr, 0, nullptr) ==
              MGPU_CONTEXT_LEASE_INVALID, "missing scope/frame cannot bind");
  MGPUImportedFrameInfo info;
  std::memset(&info, 0xff, sizeof(info));
  require(mgpuImportedFrameGetInfo(nullptr, &info, sizeof(info)) ==
              MGPU_CONTEXT_LEASE_INVALID && !info.struct_size && !info.width,
          "missing frame info clears output");
  require(mgpuImportedFrameRelease(nullptr) == MGPU_CONTEXT_LEASE_INVALID,
          "missing owner cannot release a charge");
  mgpuImportedFrameRetireAsync(nullptr, refusedRetirement, nullptr);
  mgpuImportedFrameRetireAsync(nullptr, nullptr, nullptr);
  require(callbacks == 1, "refusal callback once; null callback starts no work");
  MGPUContextLease *lease = nullptr;
  MGPUContextHandle *context = nullptr;
  require(mgpuDefaultContextRetainLease(&lease, &context) ==
              MGPU_CONTEXT_LEASE_NOT_READY && !lease && !context,
          "cold frame API cannot initialize a display context");
  require(mgpuImportedFrameGetLifetimeStats(&after, sizeof(after)) == 0 &&
          mgpuContextGetLifetimeStats(&afterContext, sizeof(afterContext)) == 0 &&
          before.live_frames == after.live_frames && before.charged_bytes == after.charged_bytes &&
          beforeContext.live_contexts == afterContext.live_contexts &&
          beforeContext.live_leases == afterContext.live_leases &&
          beforeContext.quarantined_contexts == afterContext.quarantined_contexts &&
          refs == mgpuContextRefCount(), "no cold frame/context/lease/attachment charges");
  std::cout << "PASS C/C++ imported-frame layout and cold ABI refusal/callback/accounting; "
               "no GPU initialization\n";
}

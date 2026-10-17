// Actual cold native C/C++ ABI. Never initializes graphics or capture.
#include "../include/minigpu.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <utility>

extern "C" int private_output_c_abi(void);
static void require(bool value, const char *message) {
  if (!value) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}
static int callbacks = 0;
static void refused(int result, void *userdata) {
  require(result == MGPU_CONTEXT_LEASE_INVALID && userdata == &callbacks,
          "exact refusal and caller data");
  MGPUPrivateOutputStats stats{};
  require(mgpuPrivateOutputGetStats(&stats, sizeof(stats)) == 0 &&
              stats.pending_jobs == 0 && stats.charged_bytes == 0,
          "inline callback reenters diagnostics outside native gates");
  ++callbacks;
}
template<class T> static T *sentinel() { return reinterpret_cast<T *>(uintptr_t(1)); }

int main() {
  MGPUContextLifetimeStats before{}, after{};
  require(mgpuContextGetLifetimeStats(&before, sizeof(before)) == 0,
          "creation accounting before cold refusals");
  const auto references = mgpuContextRefCount();
  require(private_output_c_abi(), "plain C layout and actual exports");
  MGPUPrivateOutputStats stats{};
  require(mgpuPrivateOutputGetStats(&stats, sizeof(stats)) == 0 &&
              stats.struct_size == 72 && stats.version == 1 &&
              stats.max_writers == 64 && stats.max_outputs == 256 &&
              stats.max_preview_loans == 256 && stats.max_descriptor_borrows == 1024 &&
              stats.max_bytes == 512ull * 1024 * 1024 &&
              stats.max_frame_bytes == 128ull * 1024 * 1024 &&
              stats.writers == 0 && stats.outputs == 0 && stats.preview_loans == 0 &&
              stats.descriptor_borrows == 0 && stats.pending_jobs == 0 &&
              stats.charged_bytes == 0,
          "versioned native quotas and empty process-rooted ownership");

  auto *lease = sentinel<MGPUContextLease>();
  auto *buffer = sentinel<MGPUBuffer>();
  auto *writer = sentinel<MGPUPrivateOutputWriter>();
  require(mgpuPrivateOutputWriterCreate(nullptr, buffer, 1, 1, &writer) ==
              MGPU_CONTEXT_LEASE_INVALID && !writer,
          "missing lease clears output without touching buffer");
  for (const auto dimensions : {std::pair<uint32_t, uint32_t>{0, 1},
                               {1, 0}, {33554433, 1},
                               {(std::numeric_limits<uint32_t>::max)(),
                                (std::numeric_limits<uint32_t>::max)()}}) {
    writer = sentinel<MGPUPrivateOutputWriter>();
    require(mgpuPrivateOutputWriterCreate(lease, buffer, dimensions.first,
                                         dimensions.second, &writer) ==
                MGPU_CONTEXT_LEASE_INVALID && !writer,
            "zero/oversized/overflow geometry refuses before owner access");
  }
  require(mgpuPrivateOutputWriterCreate(lease, buffer, 1, 1, nullptr) ==
              MGPU_CONTEXT_LEASE_INVALID &&
              mgpuPrivateOutputCreate(sentinel<MGPUPrivateOutputWriter>(), nullptr) ==
                  MGPU_CONTEXT_LEASE_INVALID,
          "output storage validation precedes native owner access");

  MGPUPrivateOutputInfo info{};
  std::memset(&info, 0x5a, sizeof(info));
  const auto untouched = info;
  require(mgpuPrivateOutputGetInfo(nullptr, &info, sizeof(info) - 1) ==
              MGPU_CONTEXT_LEASE_INVALID &&
              std::memcmp(&info, &untouched, sizeof(info)) == 0,
          "short output storage is not overwritten");
  require(mgpuPrivateOutputWriterGetInfo(nullptr, &info, sizeof(info)) ==
              MGPU_CONTEXT_LEASE_INVALID && info.struct_size == 0 &&
              mgpuPrivateOutputGetInfo(nullptr, &info, sizeof(info)) ==
                  MGPU_CONTEXT_LEASE_INVALID &&
              mgpuPrivatePreviewLoanGetInfo(nullptr, &info, sizeof(info)) ==
                  MGPU_CONTEXT_LEASE_INVALID,
          "missing owners clear full diagnostic outputs");
  auto *borrow = sentinel<MGPUPrivatePreviewBorrow>();
  MGPUPrivatePreviewBorrowInfo descriptor{};
  require(mgpuPrivatePreviewBorrowDescriptor(nullptr, &borrow, &descriptor,
                                              sizeof(descriptor)) ==
              MGPU_CONTEXT_LEASE_INVALID && !borrow &&
              descriptor.legacy_shared_handle == 0,
          "missing loan cannot expose a native descriptor");

  mgpuPrivateOutputWriteAsync(nullptr, nullptr, refused, &callbacks);
  mgpuPrivateOutputWriteAsync(sentinel<MGPUPrivateOutputWriter>(), nullptr,
                               refused, &callbacks);
  mgpuPrivateOutputWriteAsync(nullptr, sentinel<MGPUPrivateOutput>(), refused, &callbacks);
  mgpuPrivateOutputWriterRetireAsync(nullptr, refused, &callbacks);
  mgpuPrivateOutputRetireAsync(nullptr, refused, &callbacks);
  require(callbacks == 5, "one inline callback for every refused async operation");
  mgpuPrivateOutputWriteAsync(sentinel<MGPUPrivateOutputWriter>(),
                               sentinel<MGPUPrivateOutput>(), nullptr, nullptr);
  mgpuPrivateOutputWriterRetireAsync(sentinel<MGPUPrivateOutputWriter>(), nullptr, nullptr);
  mgpuPrivateOutputRetireAsync(sentinel<MGPUPrivateOutput>(), nullptr, nullptr);
  require(callbacks == 5, "null callbacks touch no native resource");
  require(mgpuPrivateOutputWriterSeal(nullptr) == MGPU_CONTEXT_LEASE_INVALID &&
              mgpuPrivateOutputSeal(nullptr) == MGPU_CONTEXT_LEASE_INVALID &&
              mgpuPrivateOutputWriterRelease(nullptr) == MGPU_CONTEXT_LEASE_INVALID &&
              mgpuPrivateOutputRelease(nullptr) == MGPU_CONTEXT_LEASE_INVALID &&
              mgpuPrivatePreviewBorrowConsumed(nullptr) == MGPU_CONTEXT_LEASE_INVALID &&
              mgpuPrivatePreviewLoanSeal(nullptr) == MGPU_CONTEXT_LEASE_INVALID &&
              mgpuPrivatePreviewLoanQuarantine(nullptr) == MGPU_CONTEXT_LEASE_INVALID &&
              mgpuPrivatePreviewLoanReleaseAfterRemoval(nullptr, 1) == MGPU_CONTEXT_LEASE_INVALID &&
              mgpuPrivatePreviewLoanReleaseAfterRemoval(sentinel<MGPUPrivatePreviewLoan>(), 0) ==
                  MGPU_CONTEXT_LEASE_INVALID,
          "absent or invalid identity cannot authorize release");
  MGPUContextLease *coldLease = nullptr;
  MGPUContextHandle *coldContext = nullptr;
  require(mgpuDefaultContextRetainLease(&coldLease, &coldContext) ==
              MGPU_CONTEXT_LEASE_NOT_READY && !coldLease && !coldContext &&
              mgpuContextGetLifetimeStats(&after, sizeof(after)) == 0 &&
              before.live_contexts == after.live_contexts &&
              before.live_leases == after.live_leases &&
              before.quarantined_contexts == after.quarantined_contexts &&
              references == mgpuContextRefCount(),
          "all refusals leave graphics cold and creation ownership unchanged");
  require(mgpuPrivateOutputGetStats(&stats, sizeof(stats)) == 0 &&
              !stats.writers && !stats.outputs && !stats.preview_loans &&
              !stats.descriptor_borrows && !stats.pending_jobs && !stats.charged_bytes,
          "refusals retain no new output/loan/job charge");
  std::cout << "PASS immutable private output cold C/C++ ABI, quotas, callback "
               "reentry and unchanged creation accounting; no GPU initialization\n";
}

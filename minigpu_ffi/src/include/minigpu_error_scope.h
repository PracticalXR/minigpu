#ifndef MINIGPU_ERROR_SCOPE_H
#define MINIGPU_ERROR_SCOPE_H
// Internal exact-device object validation. Use only on its creation worker and
// pump outside GPU locks. No foreign callbacks, timeout or success assumption.
#include "minigpu_context_owner.h"
#include <array>
#include <chrono>
#include <utility>
namespace mgpu {
// platform_sleep.h's platformWaitBackoff, out of line so this header does not
// pull <windows.h> into its includers.
void waitBackoffStep(long long waitedMicros) noexcept;
class PinnedErrorScopes {
  struct Receipt { std::atomic<bool> done{false}; bool success = false; };
  static void complete(WGPUPopErrorScopeStatus status, WGPUErrorType type,
                       WGPUStringView, void *userdata, void *) {
    auto &r = *static_cast<Receipt *>(userdata);
    r.success = status == WGPUPopErrorScopeStatus_Success &&
                type == WGPUErrorType_NoError;
    r.done.store(true, std::memory_order_release);
  }
public:
  explicit PinnedErrorScopes(std::shared_ptr<ContextPin> pin) : pin_(std::move(pin)) {
    wgpuDevicePushErrorScope(pin_->device(), WGPUErrorFilter_Validation);
    wgpuDevicePushErrorScope(pin_->device(), WGPUErrorFilter_OutOfMemory);
    wgpuDevicePushErrorScope(pin_->device(), WGPUErrorFilter_Internal);
  }
  ~PinnedErrorScopes() { if (open_) finish(); }
  bool finish() {
    if (!open_) return success_;
    for (auto &r : receipts_) {
      WGPUPopErrorScopeCallbackInfo info{};
      info.mode = WGPUCallbackMode_AllowProcessEvents;
      info.callback = complete;
      info.userdata1 = &r;
      wgpuDevicePopErrorScope(pin_->device(), info);
    }
    open_ = false;
    const auto allDone = [this] {
      for (auto &r : receipts_)
        if (!r.done.load(std::memory_order_acquire)) return false;
      return true;
    };
    // The pops nearly always resolve in the first pump: check straight after
    // it, and back off only while one is still pending.
    const auto start = std::chrono::steady_clock::now();
    while (!allDone()) {
      wgpuInstanceProcessEvents(pin_->instance());
      if (allDone()) break;
      waitBackoffStep(std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::steady_clock::now() - start)
                          .count());
    }
    success_ = true;
    for (auto &r : receipts_) success_ &= r.success;
    return success_;
  }
private:
  std::shared_ptr<ContextPin> pin_;
  std::array<Receipt, 3> receipts_{};
  bool open_ = true, success_ = false;
};
} // namespace mgpu
#endif

#ifndef MINIGPU_VIEW_PRIVATE_PREVIEW_DISPATCHER_H_
#define MINIGPU_VIEW_PRIVATE_PREVIEW_DISPATCHER_H_
#include "private_preview_owner.h"
#include <windows.h>
#include <atomic>
#include <memory>

namespace minigpu_view::private_preview {
struct Host;
// Installed into the registration before any platform/registrar call. Closing
// cannot be undone; the original deadline is checked even without a Dart timer.
struct SourceFence {
  const uint64_t epoch, deadline;
  std::atomic<bool> closed{false};
  std::atomic<bool> shutdown{false};
  std::atomic<uint32_t> registrations{0};
  SourceFence(uint64_t e, uint64_t d) : epoch(e), deadline(d) {}
  bool current() noexcept {
    if (GetTickCount64() >= deadline) closed.store(true);
    return !closed.load() && !shutdown.load();
  }
};
std::shared_ptr<Host> RetainPrivateHost(MGPUViewTrustedHost *);
bool PrivateHostPlatformCurrent(const std::shared_ptr<Host> &, bool requireRaster = true);
bool PrivateHostOpen(const std::shared_ptr<Host> &);
bool PrivateHostRegistrar(const std::shared_ptr<Host> &, flutter::TextureRegistrar *);
void *PrivateHostOriginalQuery(const std::shared_ptr<Host> &);
bool TryPrivateHostAddChild(const std::shared_ptr<Host> &);
void PrivateHostRemoveChild(const std::shared_ptr<Host> &);
int RegisterPrivateSourceOutput(const std::shared_ptr<Host> &, const std::shared_ptr<SourceFence> &,
    void *, uint64_t, MGPUViewPrivateRegistration **, MGPUViewPrivateInfo *, uint32_t);
bool PrivateRegistrationMatchesSource(MGPUViewPrivateRegistration *, const std::shared_ptr<SourceFence> &);
void QuarantinePrivateSource(const std::shared_ptr<SourceFence> &) noexcept;
}
namespace minigpu_view {
void ClosePrivatePreviewDispatchers(flutter::TextureRegistrar *) noexcept;
}
#endif

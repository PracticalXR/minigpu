#ifndef MGPU_VIEW_PRIVATE_PREVIEW_IMPL
#define MGPU_VIEW_PRIVATE_PREVIEW_IMPL
#endif
#include "registrar_association.h"
#include <windows.h>
#include <array>
#include <atomic>
#include <limits>
#include <mutex>

namespace minigpu_view {
constexpr size_t maxAssociations = 16, maxAcquisitions = 128;
struct RegistrarAssociationState {
  void *core = nullptr, *plugin = nullptr;
  flutter::TextureRegistrar *registrar = nullptr;
  uint64_t generation = 0;
  DWORD platform = 0;
  HANDLE thread = nullptr;
  std::atomic<uint32_t> phase{0}; // Unpublished reservation.
  size_t slot = 0, acquisitions = 0;
  bool owner = true, binding = false; // Registry gate.
  ~RegistrarAssociationState() { if (thread) CloseHandle(thread); }
  bool alive() const noexcept { return thread && WaitForSingleObject(thread, 0) == WAIT_TIMEOUT; }
  bool current() const noexcept { return GetCurrentThreadId() == platform && alive(); }
  bool open() const noexcept { return phase == MGPU_VIEW_ASSOCIATION_OPEN && alive(); }
};
namespace {
struct Registry {
  std::mutex gate;
  std::array<std::shared_ptr<RegistrarAssociationState>, maxAssociations> states{};
  uint64_t next = 1;
  size_t acquisitions = 0;
};
Registry &registry() { static auto *r = new Registry; return *r; }
void prune(Registry &r, const std::shared_ptr<RegistrarAssociationState> &s) {
  if (!s->owner && !s->acquisitions && !s->binding && r.states[s->slot] == s)
    r.states[s->slot].reset();
}
}
RegistrarAssociationOwner::RegistrarAssociationOwner(std::shared_ptr<RegistrarAssociationState> s)
    : state_(std::move(s)) {}
RegistrarAssociationOwner::~RegistrarAssociationOwner() { Close(); }
int RegistrarAssociationOwner::Reserve(void *core, void *plugin,
    std::unique_ptr<RegistrarAssociationOwner> &out) {
  if (out || !core || !plugin || sizeof(void *) != 8) return MGPU_VIEW_INVALID;
  try {
    auto s = std::make_shared<RegistrarAssociationState>();
    s->core = core; s->plugin = plugin; s->platform = GetCurrentThreadId();
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
        reinterpret_cast<LPCWSTR>(&MGPUViewGetRegistrarAssociationInfo), &module) ||
        !DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &s->thread,
                         THREAD_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, FALSE, 0))
      return MGPU_VIEW_NOT_READY;
    auto box = std::unique_ptr<RegistrarAssociationOwner>(new RegistrarAssociationOwner(s));
    auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate);
    for (const auto &old : r.states) if (old && old->core == core && old->phase != MGPU_VIEW_ASSOCIATION_CLOSED) {
      if (!old->current()) return MGPU_VIEW_WOULD_BLOCK;
      if (old->plugin != plugin) return MGPU_VIEW_INVALID;
      return old->phase == MGPU_VIEW_ASSOCIATION_OPEN ? MGPU_VIEW_OK : MGPU_VIEW_WOULD_BLOCK;
    }
    if (r.next == (std::numeric_limits<uint64_t>::max)()) return MGPU_VIEW_NOT_READY;
    for (size_t i = 0; i < maxAssociations; ++i) if (!r.states[i]) {
      s->slot = i; s->generation = r.next++; r.states[i] = s;
      out = std::move(box); return MGPU_VIEW_OK;
    }
    return MGPU_VIEW_NOT_READY;
  } catch (...) { return MGPU_VIEW_INTERNAL; }
}
void RegistrarAssociationOwner::Close() noexcept {
  // No foreign calls: the plugin fences this token before closing dispatchers
  // and strict registrations, then before its wrapper can be destroyed.
  const auto s = state_;
  try {
    auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate);
    s->phase = MGPU_VIEW_ASSOCIATION_CLOSED; s->owner = false; prune(r, s);
  } catch (...) { s->phase = MGPU_VIEW_ASSOCIATION_CLOSED; }
}
int PublishRegistrarAssociation(const std::shared_ptr<RegistrarAssociationState> &s,
    flutter::TextureRegistrar *registrar) {
  if (!s || !registrar) return MGPU_VIEW_INVALID;
  try {
    auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate);
    if (!s->current()) return MGPU_VIEW_WOULD_BLOCK;
    if (!s->owner || s->phase == MGPU_VIEW_ASSOCIATION_CLOSED) return MGPU_VIEW_RETIRED;
    if (s->phase == MGPU_VIEW_ASSOCIATION_OPEN)
      return s->registrar == registrar ? MGPU_VIEW_OK : MGPU_VIEW_INVALID;
    s->registrar = registrar; s->phase = MGPU_VIEW_ASSOCIATION_OPEN;
    return MGPU_VIEW_OK;
  } catch (...) { return MGPU_VIEW_INTERNAL; }
}
RegistrarAssociationBinding::RegistrarAssociationBinding(std::shared_ptr<RegistrarAssociationState> s)
    : state_(std::move(s)) {}
RegistrarAssociationBinding::~RegistrarAssociationBinding() {
  if (!reserved_) return;
  try {
    auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate);
    state_->binding = false; prune(r, state_);
  } catch (...) { /* Registry root retains the unresolved binding charge. */ }
}
bool RegistrarAssociationBinding::open() const noexcept { return state_->open(); }
uint32_t RegistrarAssociationBinding::platformThreadId() const noexcept { return state_->platform; }
void *RegistrarAssociationBinding::coreRegistrarOnPlatform() const noexcept {
  return state_->current() && state_->open() ? state_->core : nullptr;
}
flutter::TextureRegistrar *RegistrarAssociationBinding::registrarOnPlatform() const noexcept {
  return state_->current() && state_->open() ? state_->registrar : nullptr;
}
}
struct MGPUViewRegistrarAssociation {
  std::shared_ptr<minigpu_view::RegistrarAssociationState> state;
};
namespace minigpu_view {
int RetainRegistrarAssociationForClose(MGPUViewRegistrarAssociation *handle,
                                      MGPUViewRegistrarAssociation **out) {
  if (out) *out = nullptr;
  if (!handle || !out) return MGPU_VIEW_INVALID;
  try {
    auto box = std::make_unique<MGPUViewRegistrarAssociation>();
    auto s = handle->state;
    auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate);
    if (!s->open()) return MGPU_VIEW_RETIRED;
    if (!s->current() || r.acquisitions == maxAcquisitions) return MGPU_VIEW_WOULD_BLOCK;
    box->state = s; ++s->acquisitions; ++r.acquisitions;
    *out = box.release(); return MGPU_VIEW_OK;
  } catch (...) { return MGPU_VIEW_INTERNAL; }
}
bool SameRegistrarAssociation(const std::shared_ptr<RegistrarAssociationBinding> &binding,
                             MGPUViewRegistrarAssociation *handle) {
  return binding && handle && binding->state_ == handle->state;
}
int BeginRegistrarAssociationBinding(MGPUViewRegistrarAssociation *handle,
    std::shared_ptr<RegistrarAssociationBinding> &out) {
  if (!handle || out) return MGPU_VIEW_INVALID;
  try {
    auto s = handle->state;
    auto binding = std::shared_ptr<RegistrarAssociationBinding>(new RegistrarAssociationBinding(s));
    auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate);
    if (!s->open()) return MGPU_VIEW_RETIRED;
    if (!s->current()) return MGPU_VIEW_WOULD_BLOCK;
    if (s->binding) return MGPU_VIEW_WOULD_BLOCK;
    s->binding = true; binding->reserved_ = true; out = std::move(binding);
    return MGPU_VIEW_OK;
  } catch (...) { return MGPU_VIEW_INTERNAL; }
}
}
using namespace minigpu_view;
extern "C" int MGPUViewAcquireRegistrarAssociation(void *core, MGPUViewRegistrarAssociation **out) {
  if (out) *out = nullptr;
  if (!core || !out) return MGPU_VIEW_INVALID;
  try {
    auto box = std::make_unique<MGPUViewRegistrarAssociation>();
    auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate);
    for (const auto &s : r.states) if (s && s->core == core && s->open()) {
      if (!s->current() || r.acquisitions == maxAcquisitions) return MGPU_VIEW_WOULD_BLOCK;
      box->state = s; ++s->acquisitions; ++r.acquisitions;
      *out = box.release(); return MGPU_VIEW_OK;
    }
    return MGPU_VIEW_NOT_READY;
  } catch (...) { return MGPU_VIEW_INTERNAL; }
}
extern "C" int MGPUViewGetRegistrarAssociationInfo(MGPUViewRegistrarAssociation *handle,
    MGPUViewRegistrarAssociationInfo *out, uint32_t size) {
  if (!out || size < sizeof(*out)) return MGPU_VIEW_INVALID;
  *out = {};
  if (!handle) return MGPU_VIEW_INVALID;
  try {
    auto s = handle->state;
    *out = {sizeof(*out), 1, static_cast<uint32_t>(s->open() ? MGPU_VIEW_ASSOCIATION_OPEN : MGPU_VIEW_ASSOCIATION_CLOSED),
            s->platform, s->generation, 0};
    return MGPU_VIEW_OK;
  } catch (...) { return MGPU_VIEW_INTERNAL; }
}
extern "C" int MGPUViewReleaseRegistrarAssociation(MGPUViewRegistrarAssociation *handle) {
  if (!handle) return MGPU_VIEW_INVALID;
  try {
    auto s = handle->state;
    { auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate);
      --s->acquisitions; --r.acquisitions; prune(r, s); }
    delete handle; return MGPU_VIEW_OK;
  } catch (...) { return MGPU_VIEW_INTERNAL; }
}

#ifndef MGPU_VIEW_PRIVATE_PREVIEW_IMPL
#define MGPU_VIEW_PRIVATE_PREVIEW_IMPL
#endif
#include "include/minigpu_view/minigpu_view_private_preview.h"
#include "private_preview_owner.h"
#include "private_output_api.h"
#include "private_preview_dispatcher.h"
#include "registrar_association.h"
#include "private_owner_close.h"
#include <windows.h>
#include <array>
#include <atomic>
#include <limits>
#include <memory>
#include <mutex>

namespace minigpu_view::private_preview {
constexpr uint32_t maxHosts = 16, maxRegistrations = 256, maxDescriptors = 4;
constexpr uint64_t maxImageBytes = 128ull * 1024 * 1024;
constexpr uint64_t maxTicket = (uint64_t{1} << 54) - 1;
int viewResult(int native) {
  switch (static_cast<NativeResult>(native)) {
    case NativeResult::ok: return MGPU_VIEW_OK;
    case NativeResult::invalid: return MGPU_VIEW_INVALID;
    case NativeResult::notReady: return MGPU_VIEW_NOT_READY;
    case NativeResult::retired: return MGPU_VIEW_RETIRED;
    case NativeResult::wouldBlock: return MGPU_VIEW_WOULD_BLOCK;
    case NativeResult::internal: return MGPU_VIEW_INTERNAL;
    case NativeResult::unsupported: return MGPU_VIEW_UNSUPPORTED;
    case NativeResult::quarantined: return MGPU_VIEW_QUARANTINED;
    default: return MGPU_VIEW_INTERNAL;
  }
}
struct Host;
struct State;
struct Thread {
  HANDLE handle = nullptr;
  DWORD id = 0;
  ~Thread() { if (handle) CloseHandle(handle); }
  bool retain(void *source) {
    if (!source || !DuplicateHandle(GetCurrentProcess(), source, GetCurrentProcess(),
        &handle, THREAD_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, FALSE, 0)) return false;
    id = GetThreadId(handle);
    return id && GetProcessIdOfThread(handle) == GetCurrentProcessId() && alive();
  }
  bool alive() const { return handle && WaitForSingleObject(handle, 0) == WAIT_TIMEOUT; }
};
struct Host {
  flutter::TextureRegistrar *registrar = nullptr;
  std::shared_ptr<minigpu_view::RegistrarAssociationBinding> association;
  Thread platform, raster;
  Api api;
  void *originalQuery = nullptr;
  uint64_t generation = 0;
  std::atomic<uint64_t> closeRequest{0}; // Reserved under preview registry gate.
  std::atomic<bool> closeCompleted{false};
  uint32_t slot = 0;
  std::atomic<bool> closed{false};
  std::atomic<uint32_t> children{0};
  bool open() const {
    return !closed && (!association || association->open());
  }
  bool platformCurrent() const {
    return GetCurrentThreadId() == platform.id && platform.alive() && raster.alive();
  }
  bool rasterCurrent() const {
    return GetCurrentThreadId() == raster.id && raster.alive() && platform.alive();
  }
};
struct Descriptor {
  FlutterDesktopGpuSurfaceDescriptor value{};
  uint64_t ticket = 0;
  void *borrow = nullptr;
  bool consumeAttempted = false;
};
struct State {
  std::mutex gate;
  std::shared_ptr<Host> host;
  std::shared_ptr<SourceFence> source;
  std::unique_ptr<flutter::TextureVariant> variant;
  std::array<Descriptor, maxDescriptors> descriptors{};
  OutputInfo output{};
  void *loan = nullptr;
  uint64_t id = 0, loanId = 0, epoch = 0;
  uint32_t slot = 0, phase = MGPU_VIEW_CLOSED, borrows = 0;
  int result = MGPU_VIEW_OK;
  int64_t texture = -1;
  HANDLE wake = nullptr;
  bool rooted = false, open = false, unregisterAttempted = false;
  bool callbackSeen = false, sealAttempted = false, releaseAttempted = false;
  ~State() { if (wake) CloseHandle(wake); }
};
struct Registry {
  std::mutex gate;
  std::array<std::shared_ptr<Host>, maxHosts> hosts{};
  std::array<std::shared_ptr<State>, maxRegistrations> registrations{};
  uint64_t nextId = 1, nextTicket = 1;
  uint32_t hostCount = 0, registrationCount = 0;
  std::atomic<uint32_t> quarantined{0}, borrows{0};
};
Registry &registry() { static auto *r = new Registry; return *r; }
void signal(State &s) {
  ++s.epoch;
  SetEvent(s.wake); // Auto-reset signal survives collect-before-wait races.
}
bool pinModule(const void *anchor, HMODULE &module) {
  return anchor && GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
      GET_MODULE_HANDLE_EX_FLAG_PIN, reinterpret_cast<LPCWSTR>(anchor), &module);
}
template<class T> bool resolve(HMODULE m, const char *name, T &out) {
  out = reinterpret_cast<T>(GetProcAddress(m, name));
  return out != nullptr;
}
bool loadApi(const void *anchor, Api &api) {
  HMODULE module = nullptr;
  if (!pinModule(anchor, module) ||
      reinterpret_cast<const void *>(GetProcAddress(module,
          "mgpuExternalPoolQueryCapabilities")) != anchor) return false;
  return resolve(module, "mgpuPrivateOutputGetInfo", api.output) &&
      resolve(module, "mgpuPrivateOutputAcquirePreviewLoan", api.acquire) &&
      resolve(module, "mgpuPrivatePreviewLoanGetInfo", api.loan) &&
      resolve(module, "mgpuPrivatePreviewBorrowDescriptor", api.borrow) &&
      resolve(module, "mgpuPrivatePreviewBorrowConsumed", api.consume) &&
      resolve(module, "mgpuPrivatePreviewLoanSeal", api.seal) &&
      resolve(module, "mgpuPrivatePreviewLoanQuarantine", api.quarantine) &&
      resolve(module, "mgpuPrivatePreviewLoanReleaseAfterRemoval", api.release);
}
void quarantine(State &s) noexcept {
  s.open = false;
  if (s.phase != MGPU_VIEW_QUARANTINE && s.rooted) ++registry().quarantined;
  s.phase = MGPU_VIEW_QUARANTINE;
  s.result = MGPU_VIEW_QUARANTINED;
  // No destructive cleanup or allocation on an uncertain path.
  if (s.loan) {
    try { s.host->api.quarantine(s.loan); } catch (...) {}
  }
  signal(s);
}
bool reserve(const std::shared_ptr<State> &s) {
  auto &r = registry();
  std::lock_guard<std::mutex> lock(r.gate);
  if (!s->host->open() || (s->source && s->source->registrations.load() >= 2) ||
      r.registrationCount == maxRegistrations ||
      r.nextId == (std::numeric_limits<uint64_t>::max)()) return false;
  for (uint32_t i = 0; i < maxRegistrations; ++i) {
    if (!r.registrations[i]) {
      s->id = r.nextId++; s->slot = i; s->rooted = true;
      r.registrations[i] = s; ++r.registrationCount; ++s->host->children;
      if (s->source) ++s->source->registrations;
      return true;
    }
  }
  return false;
}
// Caller holds state gate. Only known-no-registration or RELEASED state may go.
void unroot(State &s) {
  auto &r = registry();
  std::lock_guard<std::mutex> lock(r.gate);
  if (!s.rooted || r.registrations[s.slot].get() != &s) return;
  r.registrations[s.slot].reset(); --r.registrationCount; --s.host->children;
  if (s.source) --s.source->registrations;
  if (s.phase == MGPU_VIEW_QUARANTINE) --r.quarantined;
  s.rooted = false;
}
uint64_t ticket(State &s, uint32_t index) {
  auto &r = registry();
  std::lock_guard<std::mutex> lock(r.gate);
  if (r.nextTicket > maxTicket) return 0; // Never wrap or reuse callback identity.
  return (r.nextTicket++ << 10) | (uint64_t(s.slot) << 2) | index;
}
MGPUViewPrivateInfo info(const State &s) {
  MGPUViewPrivateInfo out{};
  out.struct_size = sizeof(out); out.version = 1;
  out.phase = s.phase; out.last_result = s.result;
  out.registration_id = s.id; out.engine_generation = s.host->generation;
  out.output_id = s.output.id; out.loan_id = s.loanId; out.texture_id = s.texture;
  out.width = s.output.width; out.height = s.output.height;
  out.live_borrows = s.borrows; out.registered = s.texture >= 0 && !s.callbackSeen;
  out.wake_event = reinterpret_cast<uintptr_t>(s.wake); out.wake_epoch = s.epoch;
  return out;
}
void consumed(void *context) noexcept {
  try {
    const auto token = reinterpret_cast<uintptr_t>(context);
    if (!(token >> 10)) return;
    const auto index = static_cast<uint32_t>(token & 3);
    const auto slot = static_cast<uint32_t>((token >> 2) & 255);
    std::shared_ptr<State> s;
    {
      auto &r = registry();
      std::lock_guard<std::mutex> lock(r.gate);
      s = r.registrations[slot];
    }
    if (!s) return;
    std::lock_guard<std::mutex> lock(s->gate);
    auto &d = s->descriptors[index];
    // A duplicated/stale callback can never consume a newer borrow in this slot.
    if (d.ticket != token || !d.borrow || d.consumeAttempted) return;
    if (!s->host->rasterCurrent()) { quarantine(*s); return; }
    d.consumeAttempted = true;
    int result = MGPU_VIEW_INTERNAL;
    try { result = s->host->api.consume(d.borrow); } catch (...) {}
    if (result != MGPU_VIEW_OK) { quarantine(*s); return; }
    d.borrow = nullptr;
    --s->borrows; --registry().borrows;
    // DO NOT free descriptor storage: Flutter reads it after this callback.
    signal(*s);
  } catch (...) { /* Existing root retains any unacknowledged obligation. */ }
}
const FlutterDesktopGpuSurfaceDescriptor *descriptor(const std::shared_ptr<State> &s) noexcept {
  try {
    std::lock_guard<std::mutex> lock(s->gate);
    if (!s->open || !s->host->open() || (s->source && !s->source->current())) return nullptr;
    if (!s->host->rasterCurrent()) { quarantine(*s); return nullptr; }
    uint32_t index = 0;
    while (index < maxDescriptors && s->descriptors[index].borrow) ++index;
    if (index == maxDescriptors) return nullptr; // Bounded busy drop, no eviction.
    auto &d = s->descriptors[index];
    const auto id = ticket(*s, index);
    if (!id) { quarantine(*s); return nullptr; }
    BorrowInfo b{};
    void *native = nullptr;
    const auto result = s->host->api.borrow(s->loan, &native, &b, sizeof(b));
    if (native) {
      d.ticket = id; d.borrow = native; d.consumeAttempted = false;
      ++s->borrows; ++registry().borrows;
    }
    if (result == static_cast<int>(NativeResult::wouldBlock) && !native) return nullptr;
    if (result != MGPU_VIEW_OK || !native || b.size != sizeof(b) || b.version != 1 ||
        b.output_id != s->output.id || b.loan_id != s->loanId || !b.borrow_id ||
        b.width != s->output.width || b.height != s->output.height || b.format != 1 ||
        b.reserved || !b.handle || b.device_generation != s->output.device_generation) {
      quarantine(*s); return nullptr;
    }
    d.value = {};
    d.value.struct_size = sizeof(d.value);
    d.value.handle = reinterpret_cast<void *>(static_cast<uintptr_t>(b.handle));
    d.value.width = d.value.visible_width = b.width;
    d.value.height = d.value.visible_height = b.height;
    d.value.format = kFlutterDesktopPixelFormatBGRA8888;
    d.value.release_callback = consumed;
    d.value.release_context = reinterpret_cast<void *>(static_cast<uintptr_t>(id));
    // Slot reuse happens only at a later serialized raster descriptor invocation.
    // The qualified profile excludes reentry before PopulateTexture finishes.
    return &d.value;
  } catch (...) {
    try { std::lock_guard<std::mutex> lock(s->gate); quarantine(*s); } catch (...) {}
    return nullptr;
  }
}
void removed(const std::shared_ptr<State> &s) noexcept {
  try {
    std::lock_guard<std::mutex> lock(s->gate);
    if (s->phase == MGPU_VIEW_RELEASED) return;
    if (!s->unregisterAttempted || s->callbackSeen || !s->host->rasterCurrent() ||
        !s->host->open() || s->phase == MGPU_VIEW_QUARANTINE) {
      quarantine(*s); return;
    }
    s->callbackSeen = true;
    s->phase = MGPU_VIEW_REMOVED;
    signal(*s);
  } catch (...) { /* Root and callback capture remain retained. */ }
}
bool validOutput(const OutputInfo &v, uint64_t expected) {
  return v.size == sizeof(v) && v.version == 1 && v.kind == 2 && v.phase == 2 &&
      v.result == MGPU_VIEW_OK && !v.flags && v.id == expected &&
      !v.operations && !v.borrows && !v.parent &&
      v.context_generation && v.device_generation && v.width && v.height &&
      v.width <= maxImageBytes / 4 &&
      v.height <= maxImageBytes / (uint64_t(v.width) * 4) &&
      v.bytes == uint64_t(v.width) * v.height * 4;
}
} // namespace minigpu_view::private_preview

struct MGPUViewTrustedHost { std::shared_ptr<minigpu_view::private_preview::Host> state; };
struct MGPUViewPrivateRegistration { std::shared_ptr<minigpu_view::private_preview::State> state; };
using namespace minigpu_view::private_preview;

namespace {
int bindHost(void *registrar, const MGPUViewHostBinding *b, MGPUViewTrustedHost **out,
    const std::shared_ptr<minigpu_view::RegistrarAssociationBinding> &association,
    void *verifierIdentity) {
  if (out) *out = nullptr;
  if (!registrar || !b || !out || b->struct_size != sizeof(*b) || b->version != 1 ||
      b->reserved || !b->engine_generation || !b->qualification_receipt_id || !b->verify)
    return MGPU_VIEW_INVALID;
  if (b->profile != 1 || sizeof(void *) != 8) return MGPU_VIEW_UNSUPPORTED;
  try {
    auto host = std::make_shared<Host>();
    host->association = association;
    host->registrar = static_cast<flutter::TextureRegistrar *>(registrar);
    host->generation = b->engine_generation;
    host->originalQuery = b->original_minigpu_query;
    if (!host->platform.retain(b->platform_thread_handle) ||
        !host->raster.retain(b->raster_thread_handle) ||
        host->platform.id == host->raster.id || !host->platformCurrent())
      return MGPU_VIEW_UNSUPPORTED;
    if (!host->open()) return MGPU_VIEW_RETIRED;
    if (association && association->platformThreadId() != host->platform.id)
      return MGPU_VIEW_UNSUPPORTED;
    HMODULE engine = nullptr, verifier = nullptr, self = nullptr;
    if (!pinModule(b->engine_code_anchor, engine) ||
        !pinModule(reinterpret_cast<const void *>(b->verify), verifier) ||
        !pinModule(reinterpret_cast<const void *>(&MGPUViewGetPrivateStats), self) ||
        !loadApi(b->original_minigpu_query, host->api)) return MGPU_VIEW_UNSUPPORTED;
    // Native verifier is the trusted qualification authority. Do not infer it
    // from a matching code address, enum, callback observation or thread ID.
    const auto verified = b->verify(verifierIdentity, b, engine);
    // Qualification is foreign code. Closing the association during it must
    // win even though no host existed for the plugin shutdown snapshot yet.
    if (!host->open()) return MGPU_VIEW_RETIRED;
    if (verified != MGPU_VIEW_OK) return MGPU_VIEW_UNSUPPORTED;
    auto box = std::make_unique<MGPUViewTrustedHost>(); box->state = host;
    auto &r = registry();
    std::lock_guard<std::mutex> lock(r.gate);
    if (!host->open()) return MGPU_VIEW_RETIRED;
    for (uint32_t i = 0; i < maxHosts; ++i) {
      if (!r.hosts[i]) {
        host->slot = i; r.hosts[i] = host; ++r.hostCount;
        *out = box.release(); return MGPU_VIEW_OK;
      }
    }
    return MGPU_VIEW_WOULD_BLOCK;
  } catch (...) { return MGPU_VIEW_INTERNAL; }
}
}
extern "C" int MGPUViewBindTrustedHost(
    void *registrar, const MGPUViewHostBinding *b, MGPUViewTrustedHost **out) {
  return bindHost(registrar, b, out, nullptr, registrar);
}
extern "C" int MGPUViewBindAssociatedTrustedHost(
    MGPUViewRegistrarAssociation *handle, const MGPUViewHostBinding *b, MGPUViewTrustedHost **out) {
  if (out) *out = nullptr;
  if (!out || !b) return MGPU_VIEW_INVALID;
  std::shared_ptr<minigpu_view::RegistrarAssociationBinding> association;
  const auto result = minigpu_view::BeginRegistrarAssociationBinding(handle, association);
  if (result != MGPU_VIEW_OK) return result;
  auto *registrar = association->registrarOnPlatform();
  auto *core = association->coreRegistrarOnPlatform();
  if (!registrar || !core) return MGPU_VIEW_RETIRED;
  return bindHost(registrar, b, out, association, core);
}
extern "C" int MGPUViewRegisterPrivateOutput(
    MGPUViewTrustedHost *handle, void *output, uint64_t expected,
    MGPUViewPrivateRegistration **out, MGPUViewPrivateInfo *details, uint32_t size) {
  return RegisterPrivateSourceOutput(RetainPrivateHost(handle), nullptr,
                                    output, expected, out, details, size);
}
namespace minigpu_view::private_preview {
std::shared_ptr<Host> RetainPrivateHost(MGPUViewTrustedHost *h) { return h ? h->state : nullptr; }
bool PrivateHostPlatformCurrent(const std::shared_ptr<Host> &h, bool raster) {
  return h && GetCurrentThreadId() == h->platform.id && h->platform.alive() &&
      (!raster || h->raster.alive());
}
bool PrivateHostOpen(const std::shared_ptr<Host> &h) { return h && h->open(); }
bool PrivateHostRegistrar(const std::shared_ptr<Host> &h, flutter::TextureRegistrar *r) {
  return h && h->registrar == r;
}
void *PrivateHostOriginalQuery(const std::shared_ptr<Host> &h) { return h->originalQuery; }
bool TryPrivateHostAddChild(const std::shared_ptr<Host> &h) {
  auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate);
  if (!h->open() || h->closeRequest) return false;
  ++h->children; return true;
}
void PrivateHostRemoveChild(const std::shared_ptr<Host> &h) { --h->children; }
int ReservePrivateHostClose(const std::shared_ptr<Host> &h, MGPUViewRegistrarAssociation *association,
                           uint64_t request, uint32_t expectedChildren) {
  if (!h) return expectedChildren ? MGPU_VIEW_INVALID : MGPU_VIEW_OK;
  if (!minigpu_view::SameRegistrarAssociation(h->association, association)) return MGPU_VIEW_INVALID;
  auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate);
  if (h->closeRequest || h->children != expectedChildren) return MGPU_VIEW_WOULD_BLOCK;
  h->closeRequest = request; h->closed = true;
  return MGPU_VIEW_OK;
}
bool PrivateRegistrationMatchesSource(MGPUViewPrivateRegistration *r,
                                      const std::shared_ptr<SourceFence> &source) {
  return r && r->state->source == source;
}
void QuarantinePrivateSource(const std::shared_ptr<SourceFence> &source) noexcept {
  source->shutdown = true;
  source->closed = true;
  try {
    std::array<std::shared_ptr<State>, maxRegistrations> states{};
    { auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate); states = r.registrations; }
    for (const auto &s : states) if (s && s->source == source) {
      std::lock_guard<std::mutex> lock(s->gate);
      if (s->phase != MGPU_VIEW_RELEASED) quarantine(*s);
    }
  } catch (...) { /* Existing registration roots keep every obligation alive. */ }
}
int RegisterPrivateSourceOutput(const std::shared_ptr<Host> &host,
    const std::shared_ptr<SourceFence> &source, void *output, uint64_t expected,
    MGPUViewPrivateRegistration **out, MGPUViewPrivateInfo *details, uint32_t size) {
  if (out) *out = nullptr;
  if (!details || size < sizeof(*details)) return MGPU_VIEW_INVALID;
  *details = {};
  if (!host || !output || !expected || !out) return MGPU_VIEW_INVALID;
  std::shared_ptr<State> s;
  std::unique_ptr<MGPUViewPrivateRegistration> box;
  bool attempted = false;
  try {
    if (!host->open()) return MGPU_VIEW_RETIRED;
    if (source && !source->current()) return MGPU_VIEW_RETIRED;
    if (!host->platformCurrent()) return MGPU_VIEW_WOULD_BLOCK;
    s = std::make_shared<State>(); s->host = host; s->source = source;
    const auto queried = host->api.output(output, &s->output, sizeof(s->output));
    if (queried != MGPU_VIEW_OK || !validOutput(s->output, expected)) return MGPU_VIEW_INVALID;
    // A provider query can reenter platform shutdown before we have a registry
    // slot. The source's monotonic shutdown flag covers that missed snapshot.
    if (!host->open() || (source && !source->current())) return MGPU_VIEW_RETIRED;
    s->wake = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!s->wake) return MGPU_VIEW_INTERNAL;
    box = std::make_unique<MGPUViewPrivateRegistration>(); box->state = s;
    s->variant = std::make_unique<flutter::TextureVariant>(flutter::GpuSurfaceTexture(
        kFlutterDesktopGpuSurfaceTypeDxgiSharedHandle,
        [s](size_t, size_t) { return descriptor(s); }));
    if (!reserve(s)) { s->variant.reset(); return MGPU_VIEW_WOULD_BLOCK; }
    {
      std::lock_guard<std::mutex> lock(s->gate);
      if (!host->open() || (source && !source->current())) {
        s->variant.reset(); unroot(*s); return MGPU_VIEW_RETIRED;
      }
    }
    attempted = true;
    s->texture = host->registrar->RegisterTexture(s->variant.get());
    if (s->texture < 0) {
      // Qualified profile guarantees a negative ID has no retained registration.
      std::lock_guard<std::mutex> lock(s->gate);
      s->variant.reset(); unroot(*s); return MGPU_VIEW_INTERNAL;
    }
    {
      std::lock_guard<std::mutex> lock(s->gate);
      if (s->phase == MGPU_VIEW_QUARANTINE || !host->open() || (source && source->shutdown)) {
        quarantine(*s); *details = info(*s); *out = box.release();
        return MGPU_VIEW_QUARANTINED;
      }
    }
    // Registration exists before any one-time native loan can be accepted.
    void *loan = nullptr;
    const auto acquired = viewResult(host->api.acquire(output, &loan));
    bool shouldNotify = false;
    {
      std::lock_guard<std::mutex> lock(s->gate);
      s->loan = loan;
      if (s->phase == MGPU_VIEW_QUARANTINE || !host->open() || (source && source->shutdown)) {
        // Adopt and quarantine any loan returned across shutdown. An ordinary
        // Stop fence is drainable; shutdown uncertainty never becomes drainable.
        quarantine(*s);
      } else if (acquired != MGPU_VIEW_OK || !loan) {
        if (loan) quarantine(*s);
        else { s->result = acquired == MGPU_VIEW_OK ? MGPU_VIEW_INTERNAL : acquired; }
      } else {
        OutputInfo v{};
        const auto got = host->api.loan(loan, &v, sizeof(v));
        if (got != MGPU_VIEW_OK || v.size != sizeof(v) || v.version != 1 || v.kind != 3 ||
            v.phase != 2 || v.flags || v.result || !v.id || v.parent != s->output.id ||
            v.operations || v.borrows ||
            v.context_generation != s->output.context_generation ||
            v.device_generation != s->output.device_generation ||
            v.width != s->output.width || v.height != s->output.height ||
            v.bytes != s->output.bytes || !host->open() ||
            (source && source->shutdown)) quarantine(*s);
        else {
          s->loanId = v.id;
          if (source && !source->current()) {
            s->phase = MGPU_VIEW_CLOSED; s->result = MGPU_VIEW_RETIRED;
          } else { s->open = true; s->phase = MGPU_VIEW_ACTIVE; }
        }
      }
      shouldNotify = s->open;
    }
    if (shouldNotify) {
      const bool notified = host->open() && (!source || source->current()) &&
          host->registrar->MarkTextureFrameAvailable(s->texture);
      std::lock_guard<std::mutex> lock(s->gate);
      if (s->phase == MGPU_VIEW_QUARANTINE || !host->open() || (source && source->shutdown))
        quarantine(*s);
      else if (!notified) {
        s->open = false; s->phase = MGPU_VIEW_CLOSED;
        s->result = source && !source->current() ? MGPU_VIEW_RETIRED : MGPU_VIEW_INTERNAL;
      }
    }
    {
      std::lock_guard<std::mutex> lock(s->gate);
      *details = info(*s); *out = box.release(); return s->result;
    }
  } catch (...) {
    if (s && attempted) {
      std::lock_guard<std::mutex> lock(s->gate);
      quarantine(*s); *details = info(*s); *out = box.release();
      return MGPU_VIEW_QUARANTINED;
    }
    if (s) s->variant.reset(); // Break a never-admitted callback capture cycle.
    return MGPU_VIEW_INTERNAL;
  }
}
} // namespace minigpu_view::private_preview
extern "C" int MGPUViewBeginPrivateRetirement(MGPUViewPrivateRegistration *handle) {
  if (!handle) return MGPU_VIEW_INVALID;
  try {
    auto s = handle->state;
    if (!s->host->platformCurrent()) return MGPU_VIEW_WOULD_BLOCK;
    {
      std::lock_guard<std::mutex> lock(s->gate);
      if (s->phase == MGPU_VIEW_RELEASED) return MGPU_VIEW_OK;
      if (s->phase == MGPU_VIEW_QUARANTINE) return MGPU_VIEW_QUARANTINED;
      if (!s->host->open()) { quarantine(*s); return MGPU_VIEW_QUARANTINED; }
      if (s->unregisterAttempted) return MGPU_VIEW_WOULD_BLOCK;
      s->open = false;
      if (s->loan && !s->sealAttempted) {
        s->sealAttempted = true;
        int sealed = MGPU_VIEW_INTERNAL;
        try { sealed = s->host->api.seal(s->loan); } catch (...) {}
        if (sealed != MGPU_VIEW_OK) {
          quarantine(*s); return MGPU_VIEW_QUARANTINED;
        }
      }
      s->unregisterAttempted = true; s->phase = MGPU_VIEW_RETIRING;
    }
    if (!s->host->open()) {
      std::lock_guard<std::mutex> lock(s->gate);
      quarantine(*s); return MGPU_VIEW_QUARANTINED;
    }
    try { s->host->registrar->UnregisterTexture(s->texture, [s] { removed(s); }); }
    catch (...) {
      std::lock_guard<std::mutex> lock(s->gate);
      quarantine(*s); return MGPU_VIEW_QUARANTINED;
    }
    std::lock_guard<std::mutex> lock(s->gate);
    return s->phase == MGPU_VIEW_QUARANTINE ? MGPU_VIEW_QUARANTINED : MGPU_VIEW_OK;
  } catch (...) { return MGPU_VIEW_INTERNAL; }
}
extern "C" int MGPUViewCollectPrivateRetirement(
    MGPUViewPrivateRegistration *handle, MGPUViewPrivateReceipt *out, uint32_t size) {
  if (!out || size < sizeof(*out)) return MGPU_VIEW_INVALID;
  *out = {};
  if (!handle) return MGPU_VIEW_INVALID;
  try {
    auto s = handle->state;
    std::lock_guard<std::mutex> lock(s->gate);
    if (s->phase == MGPU_VIEW_QUARANTINE) return MGPU_VIEW_QUARANTINED;
    if (s->phase != MGPU_VIEW_RELEASED) {
      if (s->phase != MGPU_VIEW_REMOVED || s->borrows) return MGPU_VIEW_WOULD_BLOCK;
      if (s->loan) {
        if (s->releaseAttempted) { quarantine(*s); return MGPU_VIEW_QUARANTINED; }
        s->releaseAttempted = true;
        int result = MGPU_VIEW_INTERNAL;
        try { result = s->host->api.release(s->loan, s->loanId); } catch (...) {}
        if (result != MGPU_VIEW_OK) { quarantine(*s); return MGPU_VIEW_QUARANTINED; }
        s->loan = nullptr;
      }
      // Qualified actual removal and no native borrow work remain. Callback
      // captures independently retain safe CPU state through their own return.
      s->variant.reset();
      for (auto &d : s->descriptors) d.value = {};
      s->phase = MGPU_VIEW_RELEASED; s->result = MGPU_VIEW_OK; signal(*s);
    }
    out->struct_size = sizeof(*out); out->version = 1; out->result = MGPU_VIEW_OK;
    out->registration_id = s->id; out->engine_generation = s->host->generation;
    out->output_id = s->output.id; out->loan_id = s->loanId;
    return MGPU_VIEW_OK;
  } catch (...) { return MGPU_VIEW_INTERNAL; }
}
extern "C" int MGPUViewGetPrivateRegistrationInfo(
    MGPUViewPrivateRegistration *handle, MGPUViewPrivateInfo *out, uint32_t size) {
  if (!out || size < sizeof(*out)) return MGPU_VIEW_INVALID;
  *out = {};
  if (!handle) return MGPU_VIEW_INVALID;
  try {
    auto s = handle->state; std::lock_guard<std::mutex> lock(s->gate);
    *out = info(*s); return MGPU_VIEW_OK;
  } catch (...) { return MGPU_VIEW_INTERNAL; }
}
extern "C" int MGPUViewReleasePrivateRegistration(MGPUViewPrivateRegistration *handle) {
  if (!handle) return MGPU_VIEW_INVALID;
  try {
    auto s = handle->state;
    {
      std::lock_guard<std::mutex> lock(s->gate);
      if (s->phase != MGPU_VIEW_RELEASED || s->loan || s->borrows || s->variant)
        return MGPU_VIEW_WOULD_BLOCK;
      unroot(*s);
    }
    delete handle; return MGPU_VIEW_OK;
  } catch (...) { return MGPU_VIEW_INTERNAL; }
}
namespace {
void closeHost(const std::shared_ptr<Host> &h) {
  h->closed = true;
  std::array<std::shared_ptr<State>, maxRegistrations> states{};
  {
    auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate);
    states = r.registrations;
  }
  for (const auto &s : states) {
    if (s && s->host == h) {
      std::lock_guard<std::mutex> lock(s->gate);
      if (s->phase != MGPU_VIEW_RELEASED) quarantine(*s);
    }
  }
  h->closeCompleted = true;
}
}
extern "C" int MGPUViewCloseTrustedHost(MGPUViewTrustedHost *handle) {
  if (!handle) return MGPU_VIEW_INVALID;
  try {
    auto h = handle->state;
    if (GetCurrentThreadId() != h->platform.id || !h->platform.alive())
      return MGPU_VIEW_WOULD_BLOCK;
    closeHost(h); return MGPU_VIEW_OK;
  } catch (...) { return MGPU_VIEW_INTERNAL; }
}
namespace minigpu_view::private_preview {
int CloseRetainedPrivateHost(const std::shared_ptr<Host> &h) {
  if (!h) return MGPU_VIEW_OK;
  if (!PrivateHostPlatformCurrent(h, false)) return MGPU_VIEW_WOULD_BLOCK;
  closeHost(h); return MGPU_VIEW_OK;
}
}
extern "C" int MGPUViewReleaseTrustedHost(MGPUViewTrustedHost *handle) {
  if (!handle) return MGPU_VIEW_INVALID;
  try {
    auto h = handle->state;
    if (!h->closed || h->children || (h->closeRequest && !h->closeCompleted))
      return MGPU_VIEW_WOULD_BLOCK;
    {
      auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate);
      if (r.hosts[h->slot] != h) return MGPU_VIEW_INTERNAL;
      r.hosts[h->slot].reset(); --r.hostCount;
    }
    delete handle; return MGPU_VIEW_OK;
  } catch (...) { return MGPU_VIEW_INTERNAL; }
}
extern "C" int MGPUViewGetPrivateStats(MGPUViewPrivateStats *out, uint32_t size) {
  if (!out || size < sizeof(*out)) return MGPU_VIEW_INVALID;
  *out = {};
  try {
    auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate);
    out->struct_size = sizeof(*out); out->version = 1;
    out->hosts = r.hostCount; out->registrations = r.registrationCount;
    out->quarantined_registrations = r.quarantined.load();
    out->descriptor_borrows = r.borrows.load();
    out->max_hosts = maxHosts; out->max_registrations = maxRegistrations;
    out->max_descriptors = maxDescriptors;
    return MGPU_VIEW_OK;
  } catch (...) { return MGPU_VIEW_INTERNAL; }
}
namespace minigpu_view {
void ClosePrivatePreviewRegistrar(flutter::TextureRegistrar *registrar) noexcept {
  try {
    std::array<std::shared_ptr<Host>, maxHosts> hosts{};
    { auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate); hosts = r.hosts; }
    for (const auto &h : hosts) if (h && h->registrar == registrar) closeHost(h);
  } catch (...) { /* Existing bounded roots remain alive. */ }
}
}

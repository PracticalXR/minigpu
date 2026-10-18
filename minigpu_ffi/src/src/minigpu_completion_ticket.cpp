#include "../include/minigpu_completion_ticket.h"
#include "../include/minigpu_metadata_layout.h"
#include "../include/minigpu_module_owner.h"
#include "../include/minigpu_capture_cleanup_adapter.h"
#include <array>
#include <atomic>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>

namespace mgpu::completion {
constexpr size_t maxTickets = 128;
struct State {
  State(source_scope::MetadataAllocation allocation,
        std::shared_ptr<capture_cleanup::TicketCustody> held) noexcept
      : custody(std::move(held)), metadata(std::move(allocation)) {}
  ~State() { metadata.disposed(true); }
  std::shared_ptr<capture_cleanup::TicketCustody> custody;
  source_scope::MetadataAllocation metadata;
  std::mutex gate;
  uint64_t id = 0;
  size_t slot = 0;
  uint32_t kind = 0, nativeUsers = 0;
  bool callbackArrived = false; // Registry gate; admission precedes State::gate.
  MGPUQueueCompletionCallback callback = nullptr;
  void *userdata = nullptr;
  bool callReturned = false, callbackEntered = false, callbackReturned = false;
  bool consumed = false, collected = false;
  int result = MGPU_CONTEXT_LEASE_NOT_READY;
  std::atomic<bool> quarantined{false}, publicReleased{false};
  void quarantine() noexcept {
    quarantined = true;
    if (custody) custody->quarantine();
  }
};
struct Registry {
  std::mutex gate;
  std::array<std::shared_ptr<State>, maxTickets> tickets{};
  uint64_t nextId = 1;
};
Registry &registry() { return mgpu::detail::moduleOwner<Registry>(); }

// Finalizer is a lifetime base: derived State member destruction precedes its
// registry/global disposition, lock teardown and last ticket-custody release.
class NativeTail {
 public:
  NativeTail() = default;
  explicit NativeTail(State *state) : raw_(state), slot_(state->slot), custody_(state->custody) {}
  NativeTail(NativeTail &&other) noexcept
      : raw_(std::exchange(other.raw_, nullptr)), slot_(other.slot_), custody_(std::move(other.custody_)) {}
  NativeTail(const NativeTail &) = delete;
  ~NativeTail() {
    if (!raw_) return;
    std::shared_ptr<State> retired;
    try {
      auto &r = registry();
      std::lock_guard<std::mutex> lock(r.gate);
      auto &root = r.tickets[slot_];
      if (root.get() != raw_ || !root->nativeUsers) {
        if (root.get() == raw_) root->quarantine();
        if (custody_) custody_->quarantine();
        return;
      }
      --root->nativeUsers;
      if (!root->nativeUsers && root->publicReleased.load()) retired = std::move(root);
    } catch (...) {
      // Lock acquisition is the only throwing operation above: this native
      // user has not decremented, so its counted registry root still protects
      // raw_. shared_ptr reset and the atomic bookkeeping cannot throw.
      raw_->quarantine();
    }
  }
 private:
  State *raw_ = nullptr;
  size_t slot_ = 0;
  std::shared_ptr<capture_cleanup::TicketCustody> custody_;
};
class NativeRef : NativeTail {
 public:
  NativeRef() = default;
  explicit NativeRef(std::shared_ptr<State> s) : NativeTail(s.get()), state_(std::move(s)) {}
  NativeRef(NativeRef &&other) noexcept : NativeTail(std::move(other)), state_(std::move(other.state_)) {}
  NativeRef(const NativeRef &) = delete;
  ~NativeRef() = default;
  State *operator->() const { return state_.get(); }
  State *get() const { return state_.get(); }
  explicit operator bool() const { return !!state_; }
 private:
  std::shared_ptr<State> state_;
};
NativeRef retain(uint64_t id, bool callback = false) {
  if (!id) return {};
  auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate);
  for (auto &root : r.tickets) if (root && root->id == id) {
    if (root->publicReleased.load()) return {};
    if (callback) root->callbackArrived = true;
    ++root->nativeUsers; return NativeRef(root);
  }
  return {};
}
uint32_t phase(const State &s) {
  if (s.quarantined.load()) return MGPU_COMPLETION_QUARANTINED;
  if (s.collected) return MGPU_COMPLETION_COLLECTED;
  if (s.callReturned && s.callbackReturned) return MGPU_COMPLETION_COMPLETE;
  if (s.callbackEntered && !s.callbackReturned) return MGPU_COMPLETION_CALLBACK;
  return MGPU_COMPLETION_PENDING;
}
void completed(int result, void *userdata) noexcept {
  // This cookie is an identity, never a dereferenceable State address. IDs do
  // not wrap, so a delayed callback cannot target a recycled allocation.
  const auto id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(userdata));
  try {
    auto s = retain(id, true);
    if (!s) return;
#ifdef MINIGPU_COMPLETION_TEST_PROXY_TAIL
    extern void testCallbackAdmitted();
    testCallbackAdmitted();
#endif
    MGPUQueueCompletionCallback callback;
    void *caller;
    {
      std::lock_guard<std::mutex> lock(s->gate);
      if (s->publicReleased.load() || s->quarantined.load()) return;
      if (s->callbackEntered || !s->callback) { s->quarantine(); return; }
      s->callbackEntered = true;
      callback = s->callback; caller = s->userdata;
    }
    try { callback(result, caller); }
    catch (...) { s->quarantine(); return; }
    {
      std::lock_guard<std::mutex> lock(s->gate);
      // Actual caller callback return, not entry/result publication. No later
      // native path may inspect either caller pointer after this publication.
      s->callback = nullptr; s->userdata = nullptr;
      s->result = result; s->callbackReturned = true;
    }
#ifdef MINIGPU_COMPLETION_TEST_PROXY_TAIL
    // Compiled only into the CPU fixture, never into a distributed library.
    extern void testProxyTail();
    testProxyTail();
#endif
  } catch (...) {
    // Root is already installed; no successful receipt on an uncertain callback.
    try { auto s = retain(id); if (s) s->quarantine(); }
    catch (...) {}
  }
}
} // namespace mgpu::completion

struct MGPUCompletionTicket : mgpu::source_scope::MetadataUniqueRecord {
  explicit MGPUCompletionTicket(mgpu::source_scope::MetadataUniqueRecord storage) noexcept
      : MetadataUniqueRecord(std::move(storage)) {}
  mgpu::completion::State *state = nullptr;
  uint64_t id = 0;
};
namespace mgpu::completion {
NativeRef retain(MGPUCompletionTicket *handle) {
  if (!handle) return {};
  auto s = retain(handle->id);
  if (!s || s.get() != handle->state) return {};
  return s;
}

template<bool consumes, class Invoke>
int track(uint32_t kind, MGPUQueueCompletionCallback callback, void *userdata,
          MGPUCompletionTicket **out, Invoke invoke,
          const capture_cleanup::Family &family = {}) noexcept {
  if (out) *out = nullptr;
  if (!out || !callback) return MGPU_CONTEXT_LEASE_INVALID;
#ifdef __EMSCRIPTEN__
  return MGPU_CONTEXT_LEASE_UNSUPPORTED;
#else
  try {
    source_scope::MetadataReservations slots;
    const auto covered = family.reserveMetadata({source_scope::MetadataKind::TicketCustody,
        source_scope::MetadataKind::TicketState, source_scope::MetadataKind::TicketBox}, slots);
    if (covered != source_scope::Result::Ok) return capture_cleanup::nativeResult(covered);
    std::shared_ptr<capture_cleanup::TicketCustody> custody;
    const auto reserved = capture_cleanup::TicketCustody::create(family, slots, custody);
    if (reserved != source_scope::Result::Ok) return capture_cleanup::nativeResult(reserved);
    auto state = source_scope::makeMetadataShared<State>(slots,
        source_scope::MetadataKind::TicketState, std::move(custody));
    auto handle = source_scope::makeMetadataUnique<MGPUCompletionTicket>(slots,
        source_scope::MetadataKind::TicketBox);
    state->kind = kind; state->callback = callback; state->userdata = userdata;
    {
      auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate);
      if (r.nextId >= static_cast<uint64_t>((std::numeric_limits<uintptr_t>::max)()))
        return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
      size_t slot = maxTickets;
      for (size_t i = 0; i < maxTickets; ++i) if (!r.tickets[i]) {slot = i; break;}
      if (slot == maxTickets) return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
      state->slot = slot; state->id = r.nextId++; state->nativeUsers = 1;
      r.tickets[slot] = state;
    }
    NativeRef call(std::move(state));
    handle->state = call.get(); handle->id = call->id;
    try {
      if (call->custody) {
        const auto accepted = call->custody->accept();
        if (accepted != source_scope::Result::Ok) {
          call->publicReleased = true;
          return capture_cleanup::nativeResult(accepted);
        }
      }
      const auto cookie = reinterpret_cast<void *>(static_cast<uintptr_t>(call->id));
      const auto synchronous = invoke(completed, cookie);
      std::lock_guard<std::mutex> lock(call->gate);
      call->callReturned = true;
      if constexpr (consumes) {
        if (synchronous != MGPU_CONTEXT_LEASE_OK) {
          // Serialize refusal with callback admission, including a proxy which
          // already retained its state but has not entered State::gate yet.
          auto &r = registry(); std::lock_guard<std::mutex> admission(r.gate);
          if (call->callbackArrived) {
            call->quarantine();
            *out = handle.release(); return MGPU_CONTEXT_LEASE_QUARANTINED;
          }
          // Original checked-destroy refusal: no callback, no pointer consumption.
          call->callback = nullptr; call->userdata = nullptr;
          call->publicReleased = true;
          if (call->custody) call->custody->finished();
          return synchronous;
        }
        call->consumed = true;
      }
      *out = handle.release();
      return call->quarantined.load() ? MGPU_CONTEXT_LEASE_QUARANTINED : MGPU_CONTEXT_LEASE_OK;
    } catch (...) {
      call->quarantine();
      *out = handle.release();
      return MGPU_CONTEXT_LEASE_QUARANTINED;
    }
  } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
#endif
}
} // namespace mgpu::completion
using namespace mgpu::completion;

// MSVC /EHsc assumes directly named extern-C calls never throw. These fixed
// legacy backends are inside our failure boundary, including unexpected C++
// exceptions. An indirect C++ call keeps that boundary exception-transparent.
// Only the seven statically chosen functions below can reach this helper.
template<class Function, class... Args>
static auto callNative(Function entry, Args... args) {
  Function volatile fixed = entry;
  return fixed(args...);
}

extern "C" int mgpuPrivateOutputWriteTrackedAsync(MGPUPrivateOutputWriter *w, MGPUPrivateOutput *o,
    MGPUQueueCompletionCallback cb, void *ud, MGPUCompletionTicket **out) {
  if (out) *out = nullptr;
  if (!out || !cb) return MGPU_CONTEXT_LEASE_INVALID;
  mgpu::capture_cleanup::Family family;
  const auto prepared = mgpu::capture_cleanup::prepareWriteFamily(w, o, family);
  if (prepared != MGPU_CONTEXT_LEASE_OK) return prepared;
  return track<false>(MGPU_COMPLETION_WRITE_OUTPUT, cb, ud, out, [=](auto done, auto data) {
    callNative(&mgpu::capture_cleanup::writeOutput, w, o, done, data, family);
    return MGPU_CONTEXT_LEASE_OK;
  }, family);
}
#define TRACK_SOURCE_RETIRE(name, prepare, original, type, kind) \
extern "C" int name(type *owner, MGPUQueueCompletionCallback cb, void *ud, MGPUCompletionTicket **out) { \
  if (out) *out = nullptr; \
  if (!out || !cb) return MGPU_CONTEXT_LEASE_INVALID; \
  mgpu::capture_cleanup::Family family; \
  const auto prepared = mgpu::capture_cleanup::prepare(owner, family); \
  if (prepared != MGPU_CONTEXT_LEASE_OK) return prepared; \
  return track<false>(kind, cb, ud, out, [=](auto done, auto data) { \
    callNative(&mgpu::capture_cleanup::original, owner, done, data, family); \
    return MGPU_CONTEXT_LEASE_OK; }, family); \
}
TRACK_SOURCE_RETIRE(mgpuComputeShaderUseRetireTrackedAsync, prepareUseFamily, retireUse, MGPUComputeShaderUse, MGPU_COMPLETION_RETIRE_USE)
TRACK_SOURCE_RETIRE(mgpuImportedFrameRetireTrackedAsync, prepareFrameFamily, retireFrame, MGPUImportedFrame, MGPU_COMPLETION_RETIRE_FRAME)
TRACK_SOURCE_RETIRE(mgpuPrivateOutputRetireTrackedAsync, prepareOutputFamily, retireOutput, MGPUPrivateOutput, MGPU_COMPLETION_RETIRE_OUTPUT)
TRACK_SOURCE_RETIRE(mgpuPrivateOutputWriterRetireTrackedAsync, prepareWriterFamily, retireWriter, MGPUPrivateOutputWriter, MGPU_COMPLETION_RETIRE_WRITER)
#undef TRACK_SOURCE_RETIRE
#define TRACK_DESTROY(name, original, type, kind) \
extern "C" int name(MGPUContextLease *lease, type *owner, MGPUResourceCleanupCallback cb, void *ud, MGPUCompletionTicket **out) { \
  if (out) *out = nullptr; \
  if (!out || !cb || !lease || !owner) return MGPU_CONTEXT_LEASE_INVALID; \
  mgpu::capture_cleanup::Family family; \
  const auto prepared = mgpu::capture_cleanup::prepareFamily(lease, family); \
  if (prepared != MGPU_CONTEXT_LEASE_OK) return prepared; \
  return track<true>(kind, cb, ud, out, [=](auto done, auto data) {return callNative(&original, lease, owner, done, data, family);}, family); \
}
TRACK_DESTROY(mgpuComputeShaderDestroyTrackedAsync, mgpu::capture_cleanup::destroyShader, MGPUComputeShader, MGPU_COMPLETION_DESTROY_SHADER)
TRACK_DESTROY(mgpuBufferDestroyTrackedAsync, mgpu::capture_cleanup::destroyBuffer, MGPUBuffer, MGPU_COMPLETION_DESTROY_BUFFER)
#undef TRACK_DESTROY

extern "C" int mgpuCompletionTicketGetInfo(MGPUCompletionTicket *handle, MGPUCompletionInfo *out, uint32_t size) {
  if (!out || size < sizeof(*out)) return MGPU_CONTEXT_LEASE_INVALID;
  *out = {};
  try {
    auto s = retain(handle); if (!s) return MGPU_CONTEXT_LEASE_INVALID;
    std::lock_guard<std::mutex> lock(s->gate);
    *out = {sizeof(*out), 1, s->kind, phase(*s.get()), s->result,
        s->consumed ? 1u : 0u, s->id};
    return MGPU_CONTEXT_LEASE_OK;
  } catch (...) {return MGPU_CONTEXT_LEASE_INTERNAL;}
}
extern "C" int mgpuCompletionTicketCollect(MGPUCompletionTicket *handle, MGPUCompletionReceipt *out, uint32_t size) {
  if (!out || size < sizeof(*out)) return MGPU_CONTEXT_LEASE_INVALID;
  *out = {};
  try {
    auto s = retain(handle); if (!s) return MGPU_CONTEXT_LEASE_INVALID;
    std::lock_guard<std::mutex> lock(s->gate);
    if (s->quarantined.load()) return MGPU_CONTEXT_LEASE_QUARANTINED;
    if (s->collected) return MGPU_CONTEXT_LEASE_RETIRED;
    if (!s->callReturned || !s->callbackReturned) return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
    *out = {sizeof(*out), 1, s->kind, s->result, s->consumed ? 1u : 0u, 0, s->id};
    s->collected = true;
    return MGPU_CONTEXT_LEASE_OK;
  } catch (...) {return MGPU_CONTEXT_LEASE_INTERNAL;}
}
extern "C" int mgpuCompletionTicketRelease(MGPUCompletionTicket *handle) {
  if (!handle) return MGPU_CONTEXT_LEASE_INVALID;
  try {
    auto s = retain(handle); if (!s) return MGPU_CONTEXT_LEASE_INVALID;
    {
      std::lock_guard<std::mutex> lock(s->gate);
      if (s->quarantined.load()) return MGPU_CONTEXT_LEASE_QUARANTINED;
      if (!s->collected) return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
      s->publicReleased = true;
      if (s->custody) s->custody->finished();
    }
    mgpu::source_scope::MetadataDelete<MGPUCompletionTicket>{}(handle);
    return MGPU_CONTEXT_LEASE_OK;
  } catch (...) {return MGPU_CONTEXT_LEASE_INTERNAL;}
}
extern "C" int mgpuCompletionTicketGetStats(MGPUCompletionStats *out, uint32_t size) {
  if (!out || size < sizeof(*out)) return MGPU_CONTEXT_LEASE_INVALID;
  *out = {};
  try {
    auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate);
    uint32_t live = 0, quarantined = 0, released = 0;
    for (const auto &s : r.tickets) if (s) {
      ++live; if (s->quarantined.load()) ++quarantined;
      if (s->publicReleased.load()) ++released;
    }
    *out = {sizeof(*out), 1, live, static_cast<uint32_t>(maxTickets), quarantined, released, r.nextId};
    return MGPU_CONTEXT_LEASE_OK;
  } catch (...) {return MGPU_CONTEXT_LEASE_INTERNAL;}
}

namespace mgpu::allocation_layout {
void ticketMetadataLayout(MetadataLayoutBuilder &out) noexcept {
  out.shared<completion::State>(MetadataKind::TicketState);
  out.unique<MGPUCompletionTicket>(MetadataKind::TicketBox);
}
} // namespace mgpu::allocation_layout

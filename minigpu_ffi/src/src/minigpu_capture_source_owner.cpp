#include "../include/minigpu_capture_source_owner.h"
#include "../include/minigpu_setup_layout.h"
#include <atomic>
#include <mutex>
#include <utility>

namespace mgpu::capture_source {
using namespace source_scope;
namespace {
std::atomic<uint32_t> owners{0};
struct Slot {
  bool charged = false;
  ~Slot() { if (charged) --owners; }
};
bool reserveSlot() {
  auto value = owners.load();
  while (value < maxSources) if (owners.compare_exchange_weak(value, value + 1)) return true;
  return false;
}
}
namespace detail {
struct PreparedAnchor : setup_custody::SetupUniqueRecord {
  PreparedAnchor(setup_custody::SetupUniqueRecord record, Owner value) noexcept
      : SetupUniqueRecord(std::move(record)), owner(std::move(value)) {}
  Owner owner;
};
struct State : Slot {
  std::mutex gate;
  MGPUCaptureSourceDesc desc{};
  SourceRootOwner root;
  MGPUContextLease *lease = nullptr, *unresolvedLease = nullptr;
  Release release = nullptr;
  PreparedAnchor *anchor = nullptr;
  uint32_t phase = MGPU_CAPTURE_SOURCE_CREATING;
  bool releasing = false;
  void quarantine() noexcept {
    phase = MGPU_CAPTURE_SOURCE_QUARANTINED;
    if (root) root->domain().quarantine();
  }
};
}
uint32_t liveOwners() noexcept { return owners.load(); }
int validate(const MGPUCaptureSourceDesc *d, uint32_t size) noexcept {
  if (!d || size < sizeof(*d) || d->struct_size != sizeof(*d) || d->version != 1 ||
      !d->width || !d->height || d->recipe_revision != 1 || d->reserved ||
      !d->context_generation || !d->device_generation ||
      d->width > maxFrameBytes / 4 || d->height > maxFrameBytes / (uint64_t(d->width) * 4) ||
      d->frame_bytes != uint64_t(d->width) * d->height * 4)
    return MGPU_CONTEXT_LEASE_INVALID;
  return MGPU_CONTEXT_LEASE_OK;
}
PreparedSource::PreparedSource() = default;
PreparedSource::~PreparedSource() = default;
int PreparedSource::prepare(const MGPUCaptureSourceDesc &desc) noexcept {
  return prepareImpl(desc, {}, {}, false);
}
int PreparedSource::prepareFunded(const MGPUCaptureSourceDesc &desc,
    setup_custody::SetupTicket state, setup_custody::SetupTicket anchor) noexcept {
  return prepareImpl(desc, std::move(state), std::move(anchor), true);
}
int PreparedSource::prepareImpl(const MGPUCaptureSourceDesc &desc, setup_custody::SetupTicket stateTicket,
    setup_custody::SetupTicket anchorTicket, bool funded) noexcept {
  setup_custody::PendingSetup statePending(stateTicket), anchorPending(anchorTicket);
  if (funded && (!stateTicket || !anchorTicket)) return MGPU_CONTEXT_LEASE_INVALID;
  if (owner_ || validate(&desc, sizeof(desc)) != 0) return MGPU_CONTEXT_LEASE_INVALID;
  if (!reserveSlot()) return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
  Slot rollback; rollback.charged = true;
  try {
    auto state = funded ? setup_custody::makeSetupShared<detail::State>(stateTicket) : std::make_shared<detail::State>();
    state->charged = true; rollback.charged = false;
    state->desc = desc;
    auto anchor = funded ? setup_custody::makeSetupUnique<detail::PreparedAnchor>(anchorTicket, state) :
        setup_custody::SetupUnique<detail::PreparedAnchor>(new detail::PreparedAnchor({}, state));
    owner_ = std::move(state); anchor_ = std::move(anchor);
    return MGPU_CONTEXT_LEASE_OK;
  } catch (setup_custody::Error e) {
    return e == setup_custody::Error::Uncertain ? MGPU_CONTEXT_LEASE_QUARANTINED : MGPU_CONTEXT_LEASE_INVALID;
  } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
}
int PreparedSource::create(void *original, Create create, Release release) noexcept {
  if (!owner_ || attempted_ || !anchor_ || !original || !create || !release)
    return MGPU_CONTEXT_LEASE_INVALID;
  attempted_ = true;
  auto s = owner_;
  s->release = release; s->anchor = anchor_.release(); // Before ANY native entry.
  int result;
  try { result = create(original, s->desc, s->root, s->lease); }
  catch (...) { s->quarantine(); return MGPU_CONTEXT_LEASE_QUARANTINED; }
  if (result != 0) {
    // The fixed factory contract proves these null-output finite refusals left
    // no owner/ref. Unknown results never manufacture that proof.
    if (result >= -6 && result <= -1 && !s->root && !s->lease) {
      s->phase = MGPU_CAPTURE_SOURCE_JOINED;
      knownUnpublished_ = true;
      setup_custody::SetupDelete<detail::PreparedAnchor>{}(std::exchange(s->anchor, nullptr));
      return result;
    }
    s->quarantine(); return MGPU_CONTEXT_LEASE_QUARANTINED;
  }
  Snapshot snapshot;
  if (!s->root || !s->lease || s->root->domain().snapshot(snapshot) != Result::Ok ||
      snapshot.phase != Phase::Open || !snapshot.identity ||
      snapshot.spec.width != s->desc.width || snapshot.spec.height != s->desc.height ||
      snapshot.spec.recipeRevision != s->desc.recipe_revision || snapshot.frameBytes != s->desc.frame_bytes) {
    s->quarantine(); return MGPU_CONTEXT_LEASE_QUARANTINED;
  }
  s->phase = MGPU_CAPTURE_SOURCE_OPEN;
  return MGPU_CONTEXT_LEASE_OK;
}
MGPUContextLease *PreparedSource::borrowedLease() const noexcept {
  return owner_ && owner_->phase == MGPU_CAPTURE_SOURCE_OPEN ? owner_->lease : nullptr;
}
int getInfo(const Owner &s, MGPUCaptureSourceInfo &out) noexcept {
  out = {};
  if (!s) return MGPU_CONTEXT_LEASE_INVALID;
  try {
    std::lock_guard<std::mutex> lock(s->gate);
    Snapshot snapshot;
    if (!s->root || s->root->domain().snapshot(snapshot) != Result::Ok) return MGPU_CONTEXT_LEASE_INVALID;
    if (snapshot.phase == Phase::Poisoned) s->quarantine();
    const auto &d = s->desc;
    out = {sizeof(out), 1, s->phase, s->releasing ? 1u : 0u, snapshot.identity,
      d.context_generation, d.device_generation, d.frame_bytes, d.width, d.height, d.recipe_revision, 0};
    return MGPU_CONTEXT_LEASE_OK;
  } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
}
int seal(const Owner &s) noexcept {
  if (!s) return MGPU_CONTEXT_LEASE_INVALID;
  try {
    std::lock_guard<std::mutex> lock(s->gate);
    if (s->phase == MGPU_CAPTURE_SOURCE_QUARANTINED) return MGPU_CONTEXT_LEASE_QUARANTINED;
    if (!s->root || s->phase == MGPU_CAPTURE_SOURCE_CREATING || s->phase == MGPU_CAPTURE_SOURCE_JOINED)
      return MGPU_CONTEXT_LEASE_INVALID;
    if (s->root->seal() != Result::Ok) { s->quarantine(); return MGPU_CONTEXT_LEASE_QUARANTINED; }
    if (s->phase == MGPU_CAPTURE_SOURCE_OPEN) s->phase = MGPU_CAPTURE_SOURCE_SEALED;
    return MGPU_CONTEXT_LEASE_OK;
  } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; }
}
int tryReleaseJoined(const Owner &s, MGPUCaptureSourceJoined &out) noexcept {
  out = {};
  if (!s) return MGPU_CONTEXT_LEASE_INVALID;
  try {
    std::unique_lock<std::mutex> lock(s->gate);
    if (s->phase == MGPU_CAPTURE_SOURCE_QUARANTINED) return MGPU_CONTEXT_LEASE_QUARANTINED;
    if (s->releasing || s->phase == MGPU_CAPTURE_SOURCE_OPEN) return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
    if (!s->root || (s->phase != MGPU_CAPTURE_SOURCE_SEALED && s->phase != MGPU_CAPTURE_SOURCE_DRAINING))
      return MGPU_CONTEXT_LEASE_INVALID;
    s->phase = MGPU_CAPTURE_SOURCE_DRAINING;
    if (s->lease) {
      s->releasing = true;
      auto *lease = std::exchange(s->lease, nullptr);
      s->unresolvedLease = lease;
      lock.unlock();
      // No source/owner gate spans native ref destruction. This one-shot call's
      // unresolved marker and anchor stay live; another poll cannot retry it.
      const bool released = s->release(lease);
      lock.lock();
      s->releasing = false;
      if (!released) { s->quarantine(); return MGPU_CONTEXT_LEASE_QUARANTINED; }
      s->unresolvedLease = nullptr;
    }
    if (s->phase == MGPU_CAPTURE_SOURCE_QUARANTINED) return MGPU_CONTEXT_LEASE_QUARANTINED;
    Snapshot snapshot;
    if (s->root->domain().snapshot(snapshot) != Result::Ok) { s->quarantine(); return MGPU_CONTEXT_LEASE_QUARANTINED; }
    JoinReceipt receipt;
    const auto joined = s->root->tryJoin(receipt);
    if (joined == Result::WouldBlock) return MGPU_CONTEXT_LEASE_WOULD_BLOCK;
    if (joined != Result::Ok || s->root->release(receipt) != Result::Ok) {
      s->quarantine(); return MGPU_CONTEXT_LEASE_QUARANTINED;
    }
    out = {sizeof(out), 1, snapshot.identity, s->desc.context_generation, s->desc.device_generation};
    auto retiredRoot = std::move(s->root);
    setup_custody::SetupUnique<detail::PreparedAnchor> retiredAnchor(std::exchange(s->anchor, nullptr));
    s->phase = MGPU_CAPTURE_SOURCE_JOINED;
    lock.unlock();
    // Final payload/weak frees can invoke the original authority. No source
    // gate spans either destruction or its post-allocator receipt.
    retiredRoot.reset(); retiredAnchor.reset();
    return MGPU_CONTEXT_LEASE_OK;
  } catch (...) {
    // Mutex failures never supply successful native disposition or join.
    try { std::lock_guard<std::mutex> lock(s->gate); s->quarantine(); } catch (...) {}
    return MGPU_CONTEXT_LEASE_QUARANTINED;
  }
}
} // namespace mgpu::capture_source
namespace mgpu::allocation_layout {
Expected fundedPreparedStateExtent() noexcept {
  return setup_custody::setupSharedExtent<capture_source::detail::State>();
}
Expected preparedAnchorExtent() noexcept {
  return uniqueExtent<capture_source::detail::PreparedAnchor>();
}
void preparedSetupLayout(SetupLayoutBuilder &out) noexcept {
  out.shared<capture_source::detail::State>(MGPU_CAPTURE_SETUP_PREPARED_STATE);
  out.unique<capture_source::detail::PreparedAnchor>(MGPU_CAPTURE_SETUP_PREPARED_ANCHOR);
}
void fundedPreparedSetupLayout(FundedSetupLayoutBuilder &out) noexcept {
  out.shared<capture_source::detail::State>(MGPU_CAPTURE_SETUP_PREPARED_STATE);
  out.unique<capture_source::detail::PreparedAnchor>(MGPU_CAPTURE_SETUP_PREPARED_ANCHOR);
}
} // namespace mgpu::allocation_layout

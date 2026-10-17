// Actual Root/Prepared production types with a controlled authority. This does
// not exercise Rust's original-budget/SDK-module bridge or authenticated join.
#include "../../include/minigpu_capture_source_owner.h"
#include "../../include/minigpu_setup_layout.h"
#include "setup_allocation_probe.h"
#include "../../include/minigpu_capture_setup_boxes.h"
#include "../../include/minigpu_default_owner_slot.h"
#include <atomic>
#include <iostream>
#include <mutex>
#include <stdexcept>

namespace {
using namespace mgpu::setup_custody;
using namespace mgpu::source_scope;
using mgpu::allocation_layout::Status;
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
enum class Phase { Issued, Allocating, Allocated, Constructed, Disposed, Freeing, Freed, Cancelled, Quarantined };
struct Slot { Extent extent{}; Phase phase = Phase::Issued; uintptr_t address = 0; unsigned frees = 0; };
struct Authority {
  std::atomic<unsigned> refs{1};
  std::mutex mutex;
  std::array<Slot, 10> slots{};
  uint64_t charged = 0;
  bool okay = true, quarantined = false, rejectConstructed = false;
  unsigned beforeFree = 0, insideFree = 0, reentered = 0;
  const mgpu::capture_source::Owner *reentry = nullptr;
  std::mutex *registryMutex = nullptr;
  void checkUnlocked() noexcept {
    if (registryMutex) { if (registryMutex->try_lock()) registryMutex->unlock(); else okay = false; }
  }
  void add(size_t role, Extent extent) { slots[role].extent = extent; charged += extent.bytes; }
  static void retain(void *raw) noexcept {
    auto &a = *static_cast<Authority *>(raw); a.checkUnlocked(); a.refs.fetch_add(1);
  }
  static void release(void *raw) noexcept {
    auto &a = *static_cast<Authority *>(raw);
    a.checkUnlocked();
    if (a.refs.fetch_sub(1) <= 1) std::terminate(); // Original facade stays alive.
  }
  static uint32_t callback(void *raw, const MGPUSetupEventRecordV1 *record) noexcept {
    if (!record || record->struct_size != 48 || record->version != 1) return 0;
    return event(raw, {record->generation, record->row, record->role},
        static_cast<Event>(record->event), static_cast<uintptr_t>(record->address),
        {record->bytes, record->alignment}) ? 1u : 0u;
  }
  static bool event(void *raw, Key key, Event event, uintptr_t address, Extent extent) noexcept {
    auto &a = *static_cast<Authority *>(raw);
    a.checkUnlocked();
    // Reentry before taking the authority mutex: callback never runs under the
    // source's gate. The joined owner has already detached its root/anchor.
    if (a.reentry && key.role == 3 && event == Event::FreeReturned) {
      MGPUCaptureSourceInfo info{};
      if (mgpu::capture_source::getInfo(*a.reentry, info) != MGPU_CONTEXT_LEASE_INVALID) a.okay = false;
      ++a.reentered;
    }
    std::lock_guard<std::mutex> lock(a.mutex);
    if (a.refs.load() < 2 || key.generation != 7 || key.row || key.role >= a.slots.size()) return false;
    auto &s = a.slots[key.role];
    if (extent.bytes != s.extent.bytes || extent.alignment != s.extent.alignment) return false;
    if (event == Event::Quarantine) {
      if (!a.quarantined) { a.quarantined = true; retain(raw); } // Independent unknown-debt keeper.
      s.phase = Phase::Quarantined; return true;
    }
    if (s.phase == Phase::Quarantined) return false;
    switch (event) {
    case Event::BeginAllocation:
      if (s.phase != Phase::Issued) return false;
      s.phase = Phase::Allocating; return true;
    case Event::Allocated:
      if (s.phase != Phase::Allocating || !address || address % extent.alignment) return false;
      s.address = address; s.phase = Phase::Allocated; return true;
    case Event::Constructed:
      if (s.phase != Phase::Allocated || a.rejectConstructed) return false;
      s.phase = Phase::Constructed; return true;
    case Event::Disposed:
      if (s.phase != Phase::Constructed) return false;
      s.phase = Phase::Disposed; return true;
    case Event::BeginDeallocation:
      // Host bridge maps Allocated to constructor_rollback, Disposed to begin_free.
      if ((s.phase != Phase::Allocated && s.phase != Phase::Disposed) || address != s.address) return false;
      s.phase = Phase::Freeing; return true;
    case Event::FreeReturned:
      if (s.phase != Phase::Freeing || address != s.address || probe::insideDelete || a.charged < extent.bytes) return false;
      s.phase = Phase::Freed; ++s.frees; a.charged -= extent.bytes; return true;
    case Event::AllocationFailed:
      if (s.phase != Phase::Allocating) return false;
      s.phase = Phase::Cancelled; a.charged -= extent.bytes; return true;
    case Event::CancelUnallocated:
      if (s.phase != Phase::Issued) return false;
      s.phase = Phase::Cancelled; a.charged -= extent.bytes; return true;
    default: return false;
    }
  }
  SetupTicket ticket(size_t role, uint64_t generation = 7, uint32_t row = 0) {
    static const AuthorityOps ops{retain, release, callback};
    return SetupTicket::retain(this, &ops, {generation, row, static_cast<uint32_t>(role)}, slots[role].extent);
  }
};
thread_local Authority *active = nullptr;
void duringFree(uintptr_t address, bool after) noexcept {
  if (!active) return;
  std::lock_guard<std::mutex> lock(active->mutex);
  for (const auto &slot : active->slots) if (slot.address == address && slot.extent.bytes) {
    if (slot.phase != Phase::Freeing || active->charged < slot.extent.bytes) active->okay = false;
    if (after) ++active->insideFree; else ++active->beforeFree;
  }
}
int create(void *, const MGPUCaptureSourceDesc &d, SourceRootOwner &root, MGPUContextLease *&lease) {
  const auto result = SourceRoot::createFunded({d.width, d.height, d.recipe_revision, 0},
      active->ticket(0), active->ticket(1), root);
  if (result != Result::Ok) return MGPU_CONTEXT_LEASE_INTERNAL;
  lease = reinterpret_cast<MGPUContextLease *>(uintptr_t{1}); // Controlled CPU ownership seam.
  return 0;
}
bool release(MGPUContextLease *lease) noexcept { return lease == reinterpret_cast<MGPUContextLease *>(uintptr_t{1}); }
void realFactories() {
  Authority a;
  const auto root = mgpu::allocation_layout::fundedRootStateExtent();
  const auto prepared = mgpu::allocation_layout::fundedPreparedStateExtent();
  const auto box = mgpu::allocation_layout::uniqueExtent<SourceRoot>();
  const auto anchor = mgpu::allocation_layout::preparedAnchorExtent();
  require(root.status == Status::Known && prepared.status == Status::Known && anchor.status == Status::Known,
      "actual funded production layouts unknown");
  a.add(0, root.extent); a.add(1, box.extent); a.add(2, prepared.extent); a.add(3, anchor.extent);
  const auto total = a.charged;
  active = &a; probe::freeHook = duringFree;
  probe::state = {}; probe::state.active = true;
  mgpu::capture_source::Owner held;
  std::weak_ptr<mgpu::capture_source::detail::State> weak;
  const auto before = mgpu::capture_source::liveOwners();
  {
    mgpu::capture_source::PreparedSource source;
    MGPUCaptureSourceDesc desc{48, 1, 10, 9, 1, 0, 17, 19, 360};
    require(source.prepareFunded(desc, a.ticket(2), a.ticket(3)) == 0,
        "funded production Prepared factory refused");
    held = source.owner(); weak = held; a.reentry = &held;
    require(source.create(reinterpret_cast<void *>(uintptr_t{1}), create, release) == 0,
        "funded production Root factory refused");
    probe::state.active = false;
    require(probe::state.count == 4 && !probe::state.overflow, "four actual setup factories need exactly four allocations");
    for (const auto &s : a.slots) {
      if (!s.extent.bytes) continue;
      bool found = false;
      for (const auto &e : probe::state.entries) if (e.address == s.address && e.bytes == s.extent.bytes &&
          (s.extent.alignment > __STDCPP_DEFAULT_NEW_ALIGNMENT__ ? e.alignmentArgument == s.extent.alignment : e.alignmentArgument == 0)) found = true;
      require(found, "real factory request differs from actual complete-type funded extent");
    }
    require(mgpu::capture_source::seal(held) == 0, "production source seal refused");
    MGPUCaptureSourceJoined joined{};
    require(mgpu::capture_source::tryReleaseJoined(held, joined) == 0 && joined.source_id,
        "production CPU source did not join");
    require(a.slots[0].frees == 1 && a.slots[1].frees == 1 && a.slots[3].frees == 1 && a.reentered == 1,
        "joined Root/anchor frees missing or callback could not reenter unlocked source");
  }
  a.reentry = nullptr;
  held.reset();
  require(a.slots[2].phase == Phase::Disposed && !a.slots[2].frees && a.charged == prepared.extent.bytes &&
      mgpu::capture_source::liveOwners() == before && weak.expired(), "payload destruction refunded a retained weak block");
  auto retained = a.ticket(2);
  weak.reset();
  require(a.charged == 0 && a.beforeFree == 4 && a.insideFree == 4 && a.okay && a.refs.load() == 2,
      "actual free tails/refcounts did not settle once after underlying free");
  require(!retained.notify(Event::FreeReturned, a.slots[2].address) && a.charged == 0,
      "duplicate receipt must not refund or invalidate retained authority copies");
  retained = {};
  require(a.refs.load() == 1, "all native ticket copies should leave original authority alive");
  probe::freeHook = nullptr; active = nullptr;
  std::cout << "funded Root/Prepared requests=" << total << " root-state=" << root.extent.bytes << " root-box=" << box.extent.bytes
      << " prepared-state=" << prepared.extent.bytes << " anchor=" << anchor.extent.bytes << '\n';
}
struct Payload {
  explicit Payload(unsigned *drops) noexcept : drops(drops) {}
  ~Payload() noexcept { ++*drops; }
  unsigned *drops;
};
volatile bool constructorFault = true;
struct Throws { Throws() { if (constructorFault) throw 17; } ~Throws() noexcept = default; };
struct UniqueThrows : SetupUniqueRecord {
  explicit UniqueThrows(SetupUniqueRecord record) : SetupUniqueRecord(std::move(record)) {
    if (constructorFault) throw 23;
  }
  ~UniqueThrows() noexcept = default;
};
void emptyBoxes() {
  Authority a;
  const auto source = mgpu::allocation_layout::uniqueExtent<MGPUCaptureSource>().extent;
  const auto lease = mgpu::allocation_layout::uniqueExtent<MGPUContextLease>().extent;
  a.add(MGPU_CAPTURE_SETUP_SOURCE_BOX, source);
  a.add(MGPU_CAPTURE_SETUP_ORIGINAL_LEASE, lease); a.add(MGPU_CAPTURE_SETUP_SCOPED_LEASE, lease);
  active = &a; probe::freeHook = duringFree;
  probe::state = {}; probe::state.active = true;
  {
    auto box = makeSetupUnique<MGPUCaptureSource>(a.ticket(MGPU_CAPTURE_SETUP_SOURCE_BOX));
    auto original = makeSetupUnique<MGPUContextLease>(a.ticket(MGPU_CAPTURE_SETUP_ORIGINAL_LEASE));
    auto scoped = makeSetupUnique<MGPUContextLease>(a.ticket(MGPU_CAPTURE_SETUP_SCOPED_LEASE));
    require(!box->owner && !original->pin && !scoped->pin, "cold boxes must contain no native context refs");
    scoped->sourceManaged = true;
  }
  probe::state.active = false;
  require(probe::state.count == 3 && !probe::state.overflow && a.beforeFree == 3 && a.insideFree == 3 &&
      a.charged == 0 && a.refs.load() == 1 && a.okay, "real empty lease/source CPU boxes did not settle post-free");
  probe::freeHook = nullptr; active = nullptr;
}
void originalRetainBinding() {
  Authority a, b;
  const auto lease = mgpu::allocation_layout::uniqueExtent<MGPUContextLease>().extent;
  SetupTickets current, otherAuthority, otherRow, otherGeneration;
  for (size_t role = 0; role < current.roles.size(); ++role) {
    // Unused slots are only controlled identity fixtures, not native profiles.
    const auto extent = role == MGPU_CAPTURE_SETUP_ORIGINAL_LEASE ? lease : Extent{8, 8};
    a.add(role, extent); b.add(role, extent);
    current.roles[role] = a.ticket(role); otherAuthority.roles[role] = b.ticket(role);
    otherRow.roles[role] = a.ticket(role, 7, 1); otherGeneration.roles[role] = a.ticket(role, 8);
  }
  require(current.valid() && otherAuthority.valid() && otherRow.valid() && otherGeneration.valid(),
      "binding fixtures must each be internally coherent");
  auto original = makeSetupUnique<MGPUContextLease>(a.ticket(MGPU_CAPTURE_SETUP_ORIGINAL_LEASE));
  probe::state = {}; probe::state.active = true;
  require(matchesOriginalRetain(original.get(), current) && !matchesOriginalRetain(original.get(), otherAuthority) &&
      !matchesOriginalRetain(original.get(), otherRow) && !matchesOriginalRetain(original.get(), otherGeneration),
      "funded source accepted an original Retain from another authority/row/generation");
  MGPUContextLease legacy;
  require(!matchesOriginalRetain(&legacy, current), "legacy original lease was retroactively funded");
  probe::state.active = false;
  require(!probe::state.count, "original Retain binding check allocated");
}
struct SlotPayload;
using ReadySlot = mgpu::detail::DefaultOwnerSlot<SlotPayload>;
struct SlotPayload {
  SlotPayload(ReadySlot *slot, uint64_t generation, unsigned *drops) noexcept
      : slot(slot), generation(generation), drops(drops) {}
  ~SlotPayload() noexcept { ++*drops; slot->retire(generation); }
  ReadySlot *slot; uint64_t generation; unsigned *drops;
};
void registrySlot() {
  std::mutex mutex;
  ReadySlot slot(mutex);
  std::atomic<uint64_t> sequence{1};
  uint64_t previous = 0;
  for (bool holdWeak : {false, true}) {
    Authority a; a.registryMutex = &mutex;
    const auto extent = setupSharedExtent<SlotPayload>().extent;
    a.add(0, extent); unsigned drops = 0;
    auto acquisition = slot.acquire(sequence);
    require(acquisition.decision == ReadySlot::Decision::Create && acquisition.generation > previous,
        "one creator must own a fresh publication generation");
    const auto generation = acquisition.generation;
    require(slot.acquire(sequence).decision == ReadySlot::Decision::Busy, "concurrent creator must receive Busy");
    slot.abandon(generation + 1);
    require(slot.acquire(sequence).decision == ReadySlot::Decision::Busy, "stale abandon cleared another creator");
    auto owner = makeSetupShared<SlotPayload>(a.ticket(0), &slot, generation, &drops);
    require(!slot.publish(generation + 1, owner) && slot.publish(generation, owner), "publication must match its reservation");
    slot.retire(previous); // Old destructor cannot erase a replacement.
    auto reused = slot.acquire(sequence);
    require(reused.decision == ReadySlot::Decision::Ready && reused.owner == owner, "ready reuse changed original owner");
    reused.owner.reset();
    std::weak_ptr<SlotPayload> weak;
    if (holdWeak) weak = owner;
    owner.reset();
    require(drops == 1 && !slot.lookup(), "last payload did not clean its own default slot");
    require(a.charged == (holdWeak ? extent.bytes : 0), "default weak cleanup required another allocation/admission");
    weak.reset();
    require(a.charged == 0 && a.refs.load() == 1 && a.okay, "registry held a final callback or weak under its mutex");
    previous = generation;
  }
  sequence.store((std::numeric_limits<uint64_t>::max)() - 1);
  auto final = slot.acquire(sequence);
  require(final.decision == ReadySlot::Decision::Create && final.generation == (std::numeric_limits<uint64_t>::max)() - 1,
      "final generation differs");
  slot.abandon(final.generation);
  require(slot.acquire(sequence).decision == ReadySlot::Decision::Exhausted && sequence.load() == (std::numeric_limits<uint64_t>::max)(),
      "generation exhaustion must not wrap");
  slot.close();
  require(slot.acquire(sequence).decision == ReadySlot::Decision::Closed, "closed registry accepted a creator");
}
void failures() {
  const auto extent = setupSharedExtent<Payload>().extent;
  {
    Authority a; a.add(0, {extent.bytes + extent.alignment, extent.alignment});
    probe::state = {}; probe::state.active = true; unsigned drops = 0;
    try { (void)makeSetupShared<Payload>(a.ticket(0), &drops); require(false, "wrong extent accepted"); }
    catch (Error e) { require(e == Error::Refused, "wrong extent must refuse"); }
    probe::state.active = false;
    require(probe::state.count == 0 && a.charged == 0 && !drops, "extent refusal entered allocator");
  }
  {
    Authority a; a.add(0, extent); unsigned drops = 0;
    probe::state = {}; probe::state.active = true; probe::failNext = true;
    try { (void)makeSetupShared<Payload>(a.ticket(0), &drops); require(false, "allocator failure accepted"); }
    catch (const std::bad_alloc &) {}
    probe::state.active = false;
    require(!probe::failNext && !probe::state.count && a.charged == 0 && !drops && a.refs.load() == 1,
        "real operator new failure did not cancel its unallocated obligation");
  }
  {
    Authority a; a.add(0, setupSharedExtent<Throws>().extent);
    probe::state = {}; probe::state.active = true;
    try { (void)makeSetupShared<Throws>(a.ticket(0)); require(false, "throwing constructor accepted"); }
    catch (int value) { require(value == 17, "wrong constructor fault"); }
    probe::state.active = false;
    require(probe::state.count == 1 && probe::state.entries[0].freed && a.slots[0].frees == 1 &&
        a.charged == 0 && a.refs.load() == 1, "actual shared constructor rollback did not free once");
  }
  {
    Authority a; a.add(0, mgpu::allocation_layout::uniqueExtent<UniqueThrows>().extent);
    probe::state = {}; probe::state.active = true;
    try { (void)makeSetupUnique<UniqueThrows>(a.ticket(0)); require(false, "throwing unique constructor accepted"); }
    catch (int value) { require(value == 23, "wrong unique constructor fault"); }
    probe::state.active = false;
    require(probe::state.count == 1 && probe::state.entries[0].freed && a.slots[0].frees == 1 &&
        a.charged == 0 && a.refs.load() == 1, "actual unique constructor rollback did not free once");
  }
  {
    Authority a; a.add(0, extent); unsigned drops = 0;
    probe::state = {}; probe::state.active = true;
    try { (void)makeSetupShared<Payload>(a.ticket(0, 6), &drops); require(false, "stale generation accepted"); }
    catch (Error e) { require(e == Error::Refused, "stale generation must refuse"); }
    probe::state.active = false;
    require(!probe::state.count && a.charged == extent.bytes && a.refs.load() == 1,
        "stale identity allocated or cleared the current generation debt");
  }
  {
    Authority a; a.add(0, extent); unsigned drops = 0; a.rejectConstructed = true;
    probe::state = {}; probe::state.active = true;
    try { (void)makeSetupShared<Payload>(a.ticket(0), &drops); require(false, "rejected construction receipt accepted"); }
    catch (Error e) { require(e == Error::Uncertain, "bad receipt must quarantine"); }
    probe::state.active = false;
    require(probe::state.count == 1 && !probe::state.entries[0].freed && drops == 1 && a.quarantined &&
        a.charged == extent.bytes && a.refs.load() == 2, "unknown disposition lost block/debt/authority or destroyed twice");
  }
}
}
int main() {
  try {
    // if constexpr: a plain if leaves the Unsupported build unreachable code (C4702).
    if constexpr (!mgpu::allocation_layout::sharedLayoutSupported) {
      std::cout << "UNSUPPORTED funded setup layout; no factories\n";
    } else {
      realFactories(); emptyBoxes(); originalRetainBinding(); registrySlot(); failures();
      std::cout << "PASS funded native Root/Prepared and empty source/lease boxes, original Retain authority/row/generation binding, post-free debt, actual default-slot reserve/reuse/retire/generation, unlocked callbacks, copies/refusal/fault/unknown retention; controlled authority, no Rust bridge/Retain/Create/GPU/engine/capture\n";
    }
  } catch (const std::exception &e) { probe::state.active = false; std::cerr << "FAIL " << e.what() << '\n'; return 1; }
  catch (...) { probe::state.active = false; std::cerr << "FAIL unexpected funded exception\n"; return 1; }
}

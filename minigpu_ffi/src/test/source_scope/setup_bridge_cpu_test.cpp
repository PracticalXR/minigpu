// Test-only DLL: actual production Root/Prepared and allocator, no GPU/native
// context/Retain/Create. Other roles are UNISSUED fixture extents, not a profile.
#include "../../include/minigpu_capture_source_owner.h"
#include "../../include/minigpu_setup_layout.h"
#include "setup_allocation_probe.h"
#include <optional>
#include <stdexcept>
#ifdef _WIN32
#define PROBE_EXPORT extern "C" __declspec(dllexport)
#else
#define PROBE_EXPORT extern "C" __attribute__((visibility("default")))
#endif
using namespace mgpu::setup_custody;
using namespace mgpu::source_scope;
namespace {
using Observe = void (*)(void *, uint64_t, uint64_t, uint32_t);
struct Fixture {
  SetupTickets tickets;
  std::optional<mgpu::capture_source::PreparedSource> prepared;
  mgpu::capture_source::Owner held;
  std::weak_ptr<mgpu::capture_source::detail::State> weak;
  probe::Probe allocations;
  Observe observe = nullptr;
  void *observeContext = nullptr;
  bool created = false;
};
thread_local Fixture *active = nullptr;
void duringFree(uintptr_t address, bool after) noexcept {
  if (!active || !active->observe) return;
  for (const auto &entry : active->allocations.entries) if (entry.address && entry.address == address) {
    active->observe(active->observeContext, address, entry.bytes, after ? 1u : 0u);
    return;
  }
}
struct Scope {
  explicit Scope(Fixture *value) { active = value; probe::freeHook = duringFree; }
  ~Scope() { probe::freeHook = nullptr; active = nullptr; }
};
volatile bool failConstructor = true;
struct Throws {
  Throws() { if (failConstructor) throw 17; }
  ~Throws() noexcept = default;
};
bool layouts(uint32_t mode, MGPUSetupExtentV1 *out, uint32_t count) {
  if (!out || count != 10 || mode > 2 || !mgpu::allocation_layout::sharedLayoutSupported) return false;
  using namespace mgpu::allocation_layout;
  for (uint32_t i = 0; i < count; ++i) out[i] = {1, 1, 0}; // Unused fixture roles.
  const std::array<Expected, 4> actual{fundedRootStateExtent(), uniqueExtent<SourceRoot>(),
      fundedPreparedStateExtent(), preparedAnchorExtent()};
  for (uint32_t i = 0; i < actual.size(); ++i) {
    if (actual[i].status != Status::Known) return false;
    out[i] = {actual[i].extent.bytes, actual[i].extent.alignment, 0};
  }
  if (mode == 2) {
    const auto extent = setupSharedExtent<Throws>();
    if (extent.status != Status::Known) return false;
    out[0] = {extent.extent.bytes, extent.extent.alignment, 0};
  }
  return true;
}
bool import(const MGPUSetupAuthorityV1 *desc, uint32_t size, uint32_t mode, SetupTickets &out) {
  if (!desc || size != sizeof(*desc) || desc->struct_size != sizeof(*desc) || desc->version != 1 ||
      !desc->context || !desc->ops || !desc->ops->retain || !desc->ops->release || !desc->ops->event ||
      !desc->generation || desc->role_count != 10 || desc->reserved) return false;
  std::array<MGPUSetupExtentV1, 10> expected{};
  if (!layouts(mode, expected.data(), 10)) return false;
  for (uint32_t i = 0; i < 10; ++i) {
    const auto &actual = desc->roles[i];
    if (actual.reserved || actual.bytes != expected[i].bytes || actual.alignment != expected[i].alignment) return false;
  }
  for (uint32_t i = 0; i < 10; ++i) out.roles[i] = SetupTicket::retain(desc->context, desc->ops,
      {desc->generation, desc->row, i}, {desc->roles[i].bytes, desc->roles[i].alignment});
  return out.valid();
}
int create(void *raw, const MGPUCaptureSourceDesc &desc, SourceRootOwner &root, MGPUContextLease *&lease) {
  const auto &f = *static_cast<Fixture *>(raw);
  const auto result = SourceRoot::createFunded({desc.width, desc.height, desc.recipe_revision, 0},
      f.tickets.roles[0], f.tickets.roles[1], root);
  if (result != Result::Ok) return MGPU_CONTEXT_LEASE_INTERNAL;
  lease = reinterpret_cast<MGPUContextLease *>(uintptr_t{1}); // CPU seam only.
  return 0;
}
bool release(MGPUContextLease *lease) noexcept { return lease == reinterpret_cast<MGPUContextLease *>(uintptr_t{1}); }
}
PROBE_EXPORT int setupProbeLayout(uint32_t mode, MGPUSetupExtentV1 *out, uint32_t count) {
  return layouts(mode, out, count) ? 0 : -1;
}
PROBE_EXPORT int setupProbeBuild(const MGPUSetupAuthorityV1 *desc, uint32_t size, uint32_t mode,
    Observe observe, void *context, void **out) {
  if (!out) return -1;
  *out = nullptr;
  try {
    auto fixture = std::make_unique<Fixture>(); // Test scaffold outside measurement.
    if (!import(desc, size, mode, fixture->tickets)) return -1;
    fixture->observe = observe; fixture->observeContext = context;
    Scope scope(fixture.get());
    probe::state = {}; probe::state.active = true;
    if (mode == 2) {
      try { (void)makeSetupShared<Throws>(fixture->tickets.roles[0]); }
      catch (int value) { if (value != 17) return -1; }
      probe::state.active = false;
      // Real constructor rollback traversed the SAME Rust callback ledger.
      if (probe::state.count != 1 || !probe::state.entries[0].freed) return -1;
      return 1;
    }
    fixture->prepared.emplace();
    const MGPUCaptureSourceDesc source{48, 1, 10, 9, 1, 0, 17, 19, 360};
    if (fixture->prepared->prepareFunded(source, fixture->tickets.roles[2], fixture->tickets.roles[3]) != 0) return -1;
    if (mode == 1) {
      if (fixture->prepared->create(fixture.get(), create, release) != 0) return -1;
      fixture->created = true;
    }
    probe::state.active = false;
    if (probe::state.overflow || probe::state.count != (mode == 1 ? 4u : 2u)) return -1;
    fixture->allocations = probe::state;
    fixture->held = fixture->prepared->owner(); fixture->weak = fixture->held;
    *out = fixture.release();
    return 0;
  } catch (...) { probe::state.active = false; return -1; }
}
PROBE_EXPORT int setupProbeDropStrong(void *raw) {
  if (!raw) return -1;
  auto &fixture = *static_cast<Fixture *>(raw);
  Scope scope(&fixture);
  if (fixture.created) {
    MGPUCaptureSourceJoined joined{};
    if (mgpu::capture_source::seal(fixture.held) != 0 ||
        mgpu::capture_source::tryReleaseJoined(fixture.held, joined) != 0 || !joined.source_id) return -1;
  }
  fixture.prepared.reset(); fixture.held.reset();
  return fixture.weak.expired() ? 0 : -1;
}
PROBE_EXPORT int setupProbeDropWeak(void *raw) {
  if (!raw) return -1;
  auto &fixture = *static_cast<Fixture *>(raw);
  if (fixture.held || fixture.prepared || !fixture.weak.expired()) return -1;
  Scope scope(&fixture);
  fixture.weak.reset();
  return 0;
}
PROBE_EXPORT int setupProbeDestroy(void *raw) {
  if (!raw) return -1;
  auto *fixture = static_cast<Fixture *>(raw);
  if (fixture->held || fixture->prepared || fixture->weak.use_count()) return -1;
  delete fixture; // Test-only owners and native ticket copies, outside all gates.
  return 0;
}

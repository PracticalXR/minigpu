// Actual production owner/root lifecycle; only native pin/refcount calls are
// substituted. No engine, GPU, context, capture, pixels or windows are created.
#include "../../include/minigpu_capture_source_owner.h"
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
namespace cs = mgpu::capture_source;
using namespace mgpu::source_scope;
static void require(bool value, const char *why) {
  if (!value) { std::cerr << "FAIL " << why << '\n'; std::exit(1); }
}
static MGPUCaptureSourceDesc desc() { return {48, 1, 10, 9, 1, 0, 17, 19, 360}; }
struct Pin {
  SourceDomain source;
  SourceClaim claim;
  ~Pin() { require(claim.retire() == Result::Ok, "last actual pin alias retires its source claim"); }
};
struct Native {
  unsigned creates = 0, releases = 0;
  int result = 0;
  bool failRelease = false, throwCreate = false;
  SourceDomain source;
  std::weak_ptr<Pin> pin;
  std::function<void()> duringRelease;
};
struct Lease { Native *native; std::shared_ptr<Pin> pin; };
static int create(void *original, const MGPUCaptureSourceDesc &d,
    SourceRootOwner &root, MGPUContextLease *&out) {
  auto &native = *static_cast<Native *>(original); ++native.creates;
  if (native.result == -4) return -4; // Fixed known refusal: no owner/ref accepted.
  require(SourceRoot::create({d.width, d.height, d.recipe_revision, 0}, root) == Result::Ok, "real root creation");
  native.source = root->domain();
  auto pin = std::make_shared<Pin>(); pin->source = native.source;
  require(native.source.reserve(ResourceKind::ScopedPin, pin->claim) == Result::Ok &&
    pin->claim.accept() == Result::Ok, "actual pin role before simulated native refs");
  native.pin = pin;
  out = reinterpret_cast<MGPUContextLease *>(new Lease{&native, std::move(pin)});
  if (native.throwCreate) throw std::runtime_error("native disposition unknown after acceptance");
  return native.result;
}
static bool release(MGPUContextLease *value) noexcept {
  auto *lease = reinterpret_cast<Lease *>(value); auto &native = *lease->native;
  ++native.releases;
  if (native.duringRelease) native.duringRelease();
  if (native.failRelease) return false; // Still retained, no forged disposition.
  delete lease; return true;
}
static std::unique_ptr<cs::PreparedSource> open(Native &native) {
  auto prepared = std::make_unique<cs::PreparedSource>();
  require(prepared->prepare(desc()) == 0 && prepared->create(&native, create, release) == 0 &&
    prepared->borrowedLease() && !prepared->knownUnpublished(), "source factory publishes one borrowed lease");
  return prepared;
}
static void quotaAndRefusal() {
  const auto baseline = cs::liveOwners(); Native native;
  auto invalid = desc(); invalid.frame_bytes = 359;
  cs::PreparedSource bad;
  require(bad.prepare(invalid) == -1 && native.creates == 0 && cs::liveOwners() == baseline,
    "exact immutable geometry refuses before slot/holder/native entry");
  std::vector<std::unique_ptr<cs::PreparedSource>> pending;
  for (uint32_t i = baseline; i != maxSources; ++i) {
    auto item = std::make_unique<cs::PreparedSource>();
    require(item->prepare(desc()) == 0, "bounded metadata preparation"); pending.push_back(std::move(item));
  }
  cs::PreparedSource full;
  require(full.prepare(desc()) == -4 && native.creates == 0, "global owner cap before native refs");
  pending.clear(); require(cs::liveOwners() == baseline, "never-entered setup rolls back");
  {
    cs::PreparedSource refused; native.result = -4;
    require(refused.prepare(desc()) == 0 && refused.create(&native, create, release) == -4 &&
      !refused.borrowedLease() && refused.knownUnpublished() && native.releases == 0, "finite null-output refusal needs no native release");
    require(refused.create(&native, create, release) == -1 && native.creates == 1, "one native factory attempt per owner");
  }
  require(cs::liveOwners() == baseline, "known refused metadata rolls back after call tails");
}
static void actualTails() {
  const auto baseline = cs::liveOwners(), registered = registeredSources(); Native native;
  auto prepared = open(native); auto owner = prepared->owner();
  auto pinAlias = native.pin.lock();
  MGPUCaptureSourceInfo info{}; MGPUCaptureSourceJoined receipt{};
  require(cs::getInfo(owner, info) == 0 && info.phase == MGPU_CAPTURE_SOURCE_OPEN &&
    info.source_id && info.context_generation == 17 && info.device_generation == 19 &&
    info.frame_bytes == 360 && info.width == 10 && info.height == 9 && info.recipe_revision == 1,
    "native source identity and exact original descriptor");
  const auto identity = info.source_id;
  require(cs::tryReleaseJoined(owner, receipt) == -4 && native.releases == 0, "Seal required before lease disposition");
  SourceClaim frame, task, ticket, fence;
  require(native.source.reserve(ResourceKind::ImportedFrame, frame) == Result::Ok && frame.accept() == Result::Ok,
    "resource admitted before Seal");
  SourceOperation work; require(native.source.beginOperation(OperationPurpose::Work, work) == Result::Ok, "real work family");
  auto view = work.view();
  require(view.reserveContinuation(ContinuationKind::PrimaryTask, task) == Result::Ok && task.accept() == Result::Ok &&
    view.reserveCompletionTicket(ticket) == Result::Ok && ticket.accept() == Result::Ok, "task and ticket custody");
  require(cs::seal(owner) == 0 && cs::seal(owner) == 0, "synchronous idempotent Seal");
  SourceOperation forbidden;
  require(native.source.beginOperation(OperationPurpose::Work, forbidden) == Result::Sealed &&
    view.reserveContinuation(ContinuationKind::QueueFence, fence) == Result::Ok && fence.accept() == Result::Ok,
    "Seal refuses new work while previously admitted family may finish");
  require(cs::tryReleaseJoined(owner, receipt) == -4 && !receipt.source_id && native.releases == 1,
    "lease released once; actual pin/resource/family tails prevent join");
  pinAlias.reset();
  require(frame.retire() == Result::Ok && work.finish() == Result::Ok && task.retire() == Result::Ok &&
    fence.retire() == Result::Ok && ticket.retire() == Result::Ok, "actual admitted tails finish");
  require(cs::tryReleaseJoined(owner, receipt) == -4 && native.releases == 1, "remaining operation-view alias holds join");
  view = {};
  require(cs::tryReleaseJoined(owner, receipt) == 0 && receipt.source_id == identity &&
    receipt.context_generation == 17 && receipt.device_generation == 19 && native.releases == 1 &&
    registeredSources() == registered, "exact joined receipt after final actual tail");
  prepared.reset(); require(cs::liveOwners() == baseline + 1, "real C-call alias retains global owner quota");
  owner.reset(); require(cs::liveOwners() == baseline, "global quota refunds at final State destruction");
}
static void releaseOutsideGate() {
  Native native; auto prepared = open(native); auto owner = prepared->owner();
  auto pin = native.pin.lock(); // Keep join busy after native release.
  require(cs::seal(owner) == 0, "seal concurrency source");
  std::mutex gate; std::condition_variable condition; bool entered = false, proceed = false;
  native.duringRelease = [&] {
    MGPUCaptureSourceInfo info{};
    require(cs::getInfo(owner, info) == 0 && info.flags == 1 && cs::seal(owner) == 0,
      "native destruction runs outside owner/source gates");
    std::unique_lock<std::mutex> lock(gate); entered = true; condition.notify_all();
    condition.wait(lock, [&] { return proceed; });
  };
  int result = 99;
  std::thread worker([&] { MGPUCaptureSourceJoined out{}; result = cs::tryReleaseJoined(owner, out); });
  {
    std::unique_lock<std::mutex> lock(gate); condition.wait(lock, [&] { return entered; });
  }
  MGPUCaptureSourceJoined out{};
  require(cs::tryReleaseJoined(owner, out) == -4 && native.releases == 1, "overlapping helper poll cannot repeat native release");
  { std::lock_guard<std::mutex> lock(gate); proceed = true; } condition.notify_all();
  worker.join(); require(result == -4, "actual pin still owns source after release return");
  native.duringRelease = {}; pin.reset();
  require(cs::tryReleaseJoined(owner, out) == 0 && native.releases == 1, "later real final pin permits joined release");
}
static void retainedUncertainty() {
  // Deliberately process-retained native probes: unresolved actual refs must not
  // point at stack objects or disappear when public metadata is dropped.
  const auto baseline = cs::liveOwners();
  for (unsigned mode = 0; mode != 3; ++mode) {
    auto *native = new Native;
    auto prepared = std::make_unique<cs::PreparedSource>();
    require(prepared->prepare(desc()) == 0, "prepare uncertainty custody before native entry");
    native->throwCreate = mode == 0; native->result = mode == 1 ? -5 : 0;
    const auto result = prepared->create(native, create, release);
    auto owner = prepared->owner(); MGPUCaptureSourceJoined out{};
    if (mode == 2) {
      native->failRelease = true;
      require(result == 0 && cs::seal(owner) == 0 && cs::tryReleaseJoined(owner, out) == -7,
        "failed native release quarantines without refund");
    } else require(result == -7, "throw/mixed native output quarantines actual owners");
    require(cs::tryReleaseJoined(owner, out) == -7 && !out.source_id &&
      native->releases == (mode == 2 ? 1u : 0u), "uncertainty never retries or manufactures joined receipt");
    prepared.reset(); owner.reset();
    require(!native->pin.expired() && cs::liveOwners() == baseline + mode + 1,
      "independent preallocated anchor retains actual lease/root/quota after public Drop");
  }
}
int main() {
  quotaAndRefusal(); actualTails(); releaseOutsideGate();
  require(cs::liveOwners() == 0 && registeredSources() == 0, "successful cases leave no source/metadata debt");
  retainedUncertainty();
  std::cout << "PASS production source ownership: exact setup, bounded metadata quota, known refusal, Seal, actual pin/resource/task/ticket tails, one-shot release and retained uncertainty; no GPU\n";
}

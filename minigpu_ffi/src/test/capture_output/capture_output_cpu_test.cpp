#include "../capture_cpu_queue.h"
// Production policy/custody; substitute only native refs/actions and the queue.
// No GPU/context, capture, texture, window or presenter is constructed.
#include "../../include/minigpu_capture_output_owner.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace mgpu::source_scope;
using namespace mgpu::capture_work;
using namespace mgpu::capture_output;
using mgpu::capture_recipe::ResourceCustody;
using mgpu::capture_cleanup::GlobalTaskCustody;
static void require(bool good, const char *why) {
  if (!good) { std::cerr << "FAIL " << why << '\n'; std::exit(1); }
}
static SourceRootOwner root() {
  SourceRootOwner value;
  require(SourceRoot::create({10, 9, 1, 0}, value) == Result::Ok, "source root");
  return value;
}
static Snapshot snapshot(const SourceDomain &source) {
  Snapshot s; require(source.snapshot(s) == Result::Ok, "source snapshot"); return s;
}
static uint32_t resources(const SourceDomain &source, ResourceKind kind) {
  return snapshot(source).resources[static_cast<size_t>(kind)];
}
static void close(SourceRoot &source) {
  source.seal(); JoinReceipt receipt;
  require(source.tryJoin(receipt) == Result::Ok && source.release(receipt) == Result::Ok, "final source join");
}
struct Probe {
  Probe(SourceDomain domain, ResourceKind role, std::shared_ptr<bool> dead)
      : source(std::move(domain)), kind(role), destroyed(std::move(dead)) {}
  SourceDomain source;
  ResourceKind kind;
  std::shared_ptr<bool> destroyed;
  ~Probe() {
    require(resources(source, kind) == 1, "native/member destruction precedes source role refund");
    *destroyed = true;
  }
};
struct Owner : ResourceCustody {
  explicit Owner(ResourceCustody &&role) : ResourceCustody(std::move(role)) {}
  std::shared_ptr<Owner> packed;
  std::shared_ptr<Probe> native;
};
static std::shared_ptr<Owner> owner(const SourceDomain &source, ResourceKind kind) {
  ResourceCustody role;
  require(role.reserveSourceRole(source, kind) == Result::Ok, "role before holder/native ref");
  auto value = std::make_shared<Owner>(std::move(role));
  require(value->acceptSourceRole() == Result::Ok, "native owner accepted"); return value;
}
struct Alias : AliasCustody<Alias> {
  Alias(MetadataAllocation allocation, ResourceCustody &&role)
      : AliasCustody(std::move(allocation), std::move(role)) {}
  std::shared_ptr<Owner> parent;
  std::shared_ptr<Probe> native;
};
struct Slot { unsigned count = 0, limit = 1; bool failRefund = false; };
static bool reserveSlot(void *context) {
  auto &s = *static_cast<Slot *>(context);
  if (s.count == s.limit) return false;
  ++s.count; return true;
}
static bool refundSlot(void *context) noexcept {
  auto &s = *static_cast<Slot *>(context);
  if (!s.count || s.failRefund) return false;
  --s.count; return true;
}
static unsigned globalJobs = 0;
static SourceDomain jobSource;
static bool reserveJob() { if (globalJobs == 512) return false; ++globalJobs; return true; }
static bool refundJob() noexcept {
  Snapshot s;
  if (!globalJobs || jobSource.snapshot(s) != Result::Ok || s.operationFamilies != 1) return false;
  --globalJobs; return true;
}
struct Queue {
  bool reject = false;
  std::vector<std::function<void()>> tasks;
  static int enqueue(void *context, mgpu::task_queue::Prepared &task) noexcept {
    auto &queue = *static_cast<Queue *>(context);
    return queue.reject ? MGPU_CONTEXT_LEASE_INTERNAL
        : mgpu::capture_cpu_test::enqueueWitness(queue.tasks, task);
  }
};
int main() {
  auto source = root(); const auto domain = source->domain();
  auto foreign = root();
  require(validateWriter(domain, domain, ResourceKind::PackedBuffer, 10, 9, 360, 0x84) == Result::Ok &&
    validateWriter(domain, foreign->domain(), ResourceKind::PackedBuffer, 10, 9, 360, 0x84) == Result::Invalid &&
    validateWriter(domain, {}, ResourceKind::PackedBuffer, 10, 9, 360, 0x84) == Result::Invalid &&
    validateWriter({}, domain, ResourceKind::PackedBuffer, 10, 9, 360, 0x84) == Result::Invalid &&
    validateWriter(domain, domain, ResourceKind::Parameters, 10, 9, 360, 0x84) == Result::Invalid &&
    validateWriter(domain, domain, ResourceKind::PackedBuffer, 10, 9, 360, 0x8c) == Result::Invalid &&
    validateWriter(domain, domain, ResourceKind::PackedBuffer, 9, 10, 360, 0x84) == Result::Invalid &&
    validateOutput(domain, domain, 10, 9, 359) == Result::Invalid &&
    validateOutput(domain, foreign->domain(), 10, 9, 360) == Result::Invalid &&
    validateWriter({}, {}, ResourceKind::Count, 10, 9, 360, 0x80) == Result::Ok,
    "exact output/packed provenance, role, usage and immutable geometry before references");
  close(*foreign);
  for (auto entry : {std::pair{ResourceKind::Writer, 1u}, std::pair{ResourceKind::Output, 2u},
                    std::pair{ResourceKind::PreviewLoan, 2u}, std::pair{ResourceKind::DescriptorBorrow, 8u}}) {
    std::vector<std::shared_ptr<Owner>> owners;
    for (unsigned i = 0; i != entry.second; ++i) owners.push_back(owner(domain, entry.first));
    ResourceCustody denied;
    require(denied.reserveSourceRole(domain, entry.first) == Result::WouldBlock &&
      resources(domain, entry.first) == entry.second, "original source caps 1/2/2/8");
    for (auto &value : owners) value->proveNativeReleased();
    owners.clear();
    require(resources(domain, entry.first) == 0, "known role disposal");
  }
  {
    auto packed = owner(domain, ResourceKind::PackedBuffer);
    auto writer = owner(domain, ResourceKind::Writer); writer->packed = packed;
    auto dead = std::make_shared<bool>(false);
    writer->native = std::make_shared<Probe>(domain, ResourceKind::Writer, dead);
    packed->proveNativeReleased(); packed.reset();
    writer->proveNativeReleased(); auto tail = writer; writer.reset();
    require(!*dead && resources(domain, ResourceKind::PackedBuffer) == 1 &&
      resources(domain, ResourceKind::Writer) == 1, "writer retains packed accounting through final aliases");
    tail.reset();
    require(*dead && resources(domain, ResourceKind::PackedBuffer) == 0 &&
      resources(domain, ResourceKind::Writer) == 0, "derived members and packed aliases finish before writer role");
  }
  for (auto kind : {ResourceKind::PreviewLoan, ResourceKind::DescriptorBorrow}) {
    Slot slot;
    ResourceCustody role;
    require(role.reserveSourceRole(domain, kind) == Result::Ok, "alias role before native AddRef");
    SlotReservation quota;
    require(quota.reserve(domain, reserveSlot, refundSlot, &slot) == Result::Ok, "global slot before AddRef");
    MetadataReservations slots;
    require(role.reserveResourceMetadata(slots) == Result::Ok, "complete alias batch before heap");
    const auto stateKind = kind == ResourceKind::PreviewLoan ? MetadataKind::LoanState : MetadataKind::BorrowState;
    const auto anchorKind = kind == ResourceKind::PreviewLoan ? MetadataKind::LoanAnchor : MetadataKind::BorrowAnchor;
    auto alias = makeMetadataShared<Alias>(slots, stateKind, std::move(role));
    auto anchor = makeMetadataUnique<Alias::Anchor>(slots, anchorKind, alias);
    require(alias->acceptAlias(anchor) == Result::Ok && !anchor, "preallocated independent actual alias root");
    quota.commit();
    auto dead = std::make_shared<bool>(false);
    alias->native = std::make_shared<Probe>(domain, kind, dead);
    auto retained = alias;
    alias->native.reset(); require(*dead && slot.count == 1, "actual COM release precedes global refund");
    require(refundSlot(&slot), "known global alias disposition");
    alias->releasedAlias(); alias.reset();
    require(resources(domain, kind) == 1, "proof and public release cannot drop a retained State alias");
    retained.reset(); require(resources(domain, kind) == 0, "final alias releases source role");
  }
  Slot slot;
  {
    SlotReservation quota;
    require(quota.reserve(domain, reserveSlot, refundSlot, &slot) == Result::Ok, "pre-native quota");
    SlotReservation denied;
    require(denied.reserve(domain, reserveSlot, refundSlot, &slot) == Result::WouldBlock,
      "quota refusal makes no native owner");
  }
  require(!slot.count, "ordinary pre-native rollback refunds once");
  close(*source);

  // The same global job follows the actual primary/fence/wait closures. A
  // known initial rejection may roll back; a nested failure after mutation may not.
  for (unsigned failure = 0; failure != 3; ++failure) {
    source = root(); jobSource = source->domain(); Queue queue;
    bool mutated = false; unsigned callbacks = 0;
    const auto jobsBefore = globalJobs;
    std::weak_ptr<int> nativeOwner;
    {
      Family family; require(Family::beginWork(jobSource, family) == Result::Ok, "output Work family");
      std::shared_ptr<GlobalTaskCustody> global;
      require(GlobalTaskCustody::create(family, reserveJob, refundJob, global) == Result::Ok, "one global job");
      auto actualOwner = std::make_shared<int>(1); nativeOwner = actualOwner;
      auto wait = std::make_shared<PreparedWait>();
      require(wait->prepare(family, [](WaitSignal &signal) {
        signal.completed = true; signal.result = 0; return Outcome::done();
      }, [&, family, actualOwner](int result) {
        require(result == 0 && *actualOwner == 1, "successful fence before failed EndAccess");
        completeWrite(family, MGPU_CONTEXT_LEASE_QUARANTINED, [&](int result) {
          require(result == MGPU_CONTEXT_LEASE_QUARANTINED, "caller receives failed EndAccess"); ++callbacks;
        });
      }, global) == 0,
        "wait/fence reserved before mutation");
      PreparedWork job;
      require(job.reserve(family, ContinuationKind::PrimaryTask, global) == 0 &&
        job.configure([&, wait, actualOwner] {
          require(*actualOwner == 1, "actual native owner retained by action");
          mutated = true; queue.reject = failure == 1;
          return wait->enqueue(Queue::enqueue, &queue) == 0 ? Outcome::done() : Outcome::uncertain();
        }, [&](int result) { if (result != 0) ++callbacks; }) == 0, "prepared primary");
      queue.reject = failure == 0;
      require(job.enqueue(Queue::enqueue, &queue) == (failure ? 0 : MGPU_CONTEXT_LEASE_INTERNAL),
        "known initial enqueue result");
      if (failure) {
        auto primary = std::move(queue.tasks.front()); queue.tasks.clear(); primary();
        if (failure == 2) queue.tasks.front()();
      }
    }
    queue.tasks.clear();
    if (failure) {
      JoinReceipt denied;
      const auto s = snapshot(jobSource);
      require(mutated && callbacks == 1 && globalJobs == jobsBefore + 1 && !nativeOwner.expired() &&
        source->tryJoin(denied) == Result::Poisoned && s.continuations[0] == (failure == 1 ? 1u : 0u) &&
        s.continuations[1] == 1 && s.continuations[2] == 1,
        "accepted uncertainty retains actual job/owners/global slot, never retries");
    } else {
      require(!mutated && !callbacks && globalJobs == jobsBefore && nativeOwner.expired() &&
        !snapshot(jobSource).operationFamilies,
        "known initial rejection refunds without invoking native action");
      close(*source);
    }
  }
  // This uncertain owner intentionally remains rooted; its separate native ref
  // cannot disappear with a lost public wrapper or a mere accounting poison.
  source = root(); auto uncertainDomain = source->domain();
  ResourceCustody role;
  require(role.reserveSourceRole(uncertainDomain, ResourceKind::PreviewLoan) == Result::Ok, "uncertain alias role");
  MetadataReservations slots;
  require(role.reserveResourceMetadata(slots) == Result::Ok, "uncertain alias prepaid batch");
  auto alias = makeMetadataShared<Alias>(slots, MetadataKind::LoanState, std::move(role));
  auto anchor = makeMetadataUnique<Alias::Anchor>(slots, MetadataKind::LoanAnchor, alias);
  require(alias->acceptAlias(anchor) == Result::Ok, "uncertain alias rooted before native ref");
  auto dead = std::make_shared<bool>(false);
  alias->native = std::make_shared<Probe>(uncertainDomain, ResourceKind::PreviewLoan, dead);
  std::weak_ptr<Alias> retained = alias;
  alias->quarantineAlias(); alias.reset();
  JoinReceipt denied;
  require(!retained.expired() && !*dead && source->tryJoin(denied) == Result::Poisoned &&
    resources(uncertainDomain, ResourceKind::PreviewLoan) == 1 &&
    snapshot(uncertainDomain).metadataAllocations[static_cast<size_t>(MetadataKind::LoanState)] == 1 &&
    snapshot(uncertainDomain).metadataAllocations[static_cast<size_t>(MetadataKind::LoanAnchor)] == 1,
    "independent uncertain alias anchor retains actual ref and allocation custody");
  std::cout << "PASS output policy/custody: exact source/role/geometry, caps 1/2/2/8, packed aliases, final member disposal, independent alias roots, slot rollback, primary/fence/wait global tails and accepted uncertainty; CPU only\n";
}

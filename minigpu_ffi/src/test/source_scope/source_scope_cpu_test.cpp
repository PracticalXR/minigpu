#include "../../include/minigpu_source_scope_owner.h"
#include "../../include/minigpu_creation_identity.h"
#include "../../include/minigpu_source_metadata_allocator.h"
#include "../../include/minigpu_capture_cleanup_owner.h"
#include "../../include/minigpu_metadata_layout.h"
#include <array>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace mgpu::source_scope {
std::array<std::atomic<uint32_t>, metadataKinds> allocationAttempts{};
std::array<std::atomic<size_t>, metadataKinds> allocationBytes{}, allocationAlignments{};
std::atomic<int> failMetadataKind{-1};
std::function<void(MetadataKind)> metadataAllocateHook, metadataDeallocateHook;
void testMetadataBeforeAllocate(MetadataKind kind, size_t bytes, size_t alignment) {
  const auto index = static_cast<size_t>(kind);
  ++allocationAttempts[index];
  allocationBytes[index] = bytes;
  allocationAlignments[index] = alignment;
  if (metadataAllocateHook) metadataAllocateHook(kind);
  int expected = static_cast<int>(kind);
  if (failMetadataKind.compare_exchange_strong(expected, -1)) throw std::bad_alloc();
}
void testMetadataBeforeDeallocate(MetadataKind kind) noexcept {
  if (metadataDeallocateHook) metadataDeallocateHook(kind);
}
} // namespace mgpu::source_scope

namespace {
using namespace mgpu::source_scope;
void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
void ok(Result value, const char *message) { require(value == Result::Ok, message); }
SourceRootOwner create(SourceSpec spec = {10, 9, 1, 0}) {
  SourceRootOwner result;
  ok(SourceRoot::create(spec, result), "create source");
  require(result != nullptr, "created root missing");
  return result;
}
Snapshot snapshot(const SourceDomain &domain) {
  Snapshot result;
  ok(domain.snapshot(result), "snapshot");
  return result;
}
void close(SourceRootOwner &root) {
  ok(root->seal(), "seal");
  JoinReceipt joined;
  ok(root->tryJoin(joined), "join");
  ok(root->release(joined), "release joined source");
  require(!joined, "release consumes join receipt");
  root.reset();
}
void concurrentColdRegistry() {
  // First production registry access is raced, not a helper-only initializer.
  std::array<SourceRootOwner, 8> roots;
  std::array<Result, 8> results{};
  std::array<std::thread, 8> threads;
  std::atomic<unsigned> ready{0};
  std::atomic<bool> start{false};
  for (size_t i = 0; i < threads.size(); ++i) threads[i] = std::thread([&, i] {
    ready.fetch_add(1, std::memory_order_release);
    while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
    results[i] = SourceRoot::create({10, 9, 1, 0}, roots[i]);
  });
  while (ready.load(std::memory_order_acquire) != threads.size()) std::this_thread::yield();
  start.store(true, std::memory_order_release);
  for (auto &thread : threads) thread.join();
  require(registeredSources() == roots.size(), "concurrent cold registry lost a source root");
  for (size_t i = 0; i < roots.size(); ++i) {
    ok(results[i], "concurrent cold source creation");
    require(roots[i] != nullptr, "concurrent cold source owner missing");
    const auto identity = snapshot(roots[i]->domain()).identity;
    require(identity != 0, "concurrent cold source identity missing");
    for (size_t j = 0; j < i; ++j)
      require(identity != snapshot(roots[j]->domain()).identity,
              "concurrent cold registry recycled identity");
  }
  for (auto &root : roots) close(root);
  require(registeredSources() == 0, "cold registry retained a known joined owner");
}
void geometryAndInventory() {
  const auto initial = registeredSources();
  SourceRootOwner invalid;
  for (auto spec : std::array<SourceSpec, 5>{{
           {0, 9, 1, 0}, {1, 0, 1, 0}, {10, 9, 2, 0}, {10, 9, 1, 1},
           {std::numeric_limits<uint32_t>::max(),
            std::numeric_limits<uint32_t>::max(), 1, 0}}}) {
    require(SourceRoot::create(spec, invalid) == Result::Invalid && !invalid,
            "invalid or overflowing spec must refuse without publication");
  }
  require(registeredSources() == initial, "invalid specs charged a root");
  auto bound = create({8192, 4096, 1, 0});
  require(snapshot(bound->domain()).frameBytes == maxFrameBytes,
          "exact frame-byte bound");
  close(bound);
  auto root = create();
  auto domain = root->domain();
  const auto identity = snapshot(domain).identity;
  std::vector<SourceClaim> resources;
  const std::array<uint32_t, resourceKinds> caps{1, 1, 1, 1, 1, 2, 1, 2, 8, 1};
  SourceClaim refused;
  for (size_t i = 0; i < resourceKinds; ++i) {
    for (uint32_t n = 0; n < caps[i]; ++n) {
      SourceClaim claim;
      ok(domain.reserve(static_cast<ResourceKind>(i), claim), "reserve role");
      ok(claim.accept(), "accept role");
      require(claim.cancelUnaccepted() == Result::Invalid,
              "accepted native entry cannot roll back");
      resources.push_back(std::move(claim));
    }
    require(domain.reserve(static_cast<ResourceKind>(i), refused) == Result::WouldBlock &&
                !refused, "role overflow must refuse");
  }
  auto full = snapshot(domain);
  require(full.resources == caps && full.frameBytes == 360 &&
              full.payloadBytes == 1816 && full.payloadLimit == 1816,
          "fixed five-frame-plus-parameters inventory");
  require(domain.reserve(ResourceKind::Count, refused) == Result::Invalid,
          "invalid role");
  require(snapshot(domain).payloadBytes == full.payloadBytes,
          "refusal changed accounting");
  ok(root->seal(), "seal populated source");
  require(domain.reserve(ResourceKind::Output, refused) == Result::Sealed,
          "sealed source allocated backing");
  JoinReceipt joined;
  require(root->tryJoin(joined) == Result::WouldBlock && !joined,
          "join passed live roles");
  for (auto &claim : resources) ok(claim.retire(), "retire final role");
  ok(root->tryJoin(joined), "join retired source");
  JoinReceipt duplicate;
  require(root->tryJoin(duplicate) == Result::Joined && !duplicate,
          "join minted a second receipt");
  auto other = create();
  require(other->release(joined) == Result::Invalid && joined,
          "foreign join receipt released a source");
  close(other);
  ok(root->release(joined), "release exact receipt");
  root.reset();
  require(snapshot(domain).phase == Phase::Joined &&
              domain.reserve(ResourceKind::Output, refused) == Result::Joined,
          "stale domain became a fresh or legacy allocator");
  auto replacement = create();
  require(snapshot(replacement->domain()).identity > identity,
          "source identity was recycled");
  close(replacement);
  require(registeredSources() == initial, "known cleanup leaked registry capacity");
}
void domainsAliasesAndAtomicCapacity() {
  auto root = create();
  auto other = create();
  auto domain = root->domain();
  require(sameSourceDomain(SourceDomain{}, SourceDomain{}) && sameSourceDomain(domain, root->domain()) &&
              !sameSourceDomain(domain, other->domain()) &&
              !sameSourceDomain(domain, {}) && !sameSourceDomain({}, domain),
          "source equality collapsed same-spec roots or scoped/legacy domains");
  SourceClaim pin;
  ok(domain.reserve(ResourceKind::ScopedPin, pin), "reserve pin");
  ok(pin.accept(), "accept pin");
  auto publicOwner = std::make_shared<SourceClaim>(std::move(pin));
  auto nativeAlias = publicOwner;
  publicOwner.reset();
  ok(root->seal(), "seal with pin alias");
  JoinReceipt joined;
  require(root->tryJoin(joined) == Result::WouldBlock,
          "public release refunded retained pin alias");
  ok(nativeAlias->retire(), "retire final pin alias");
  nativeAlias.reset();
  ok(root->tryJoin(joined), "join after pin disposition");
  ok(root->release(joined), "release pin source");
  root.reset();
  close(other);

  root = create();
  domain = root->domain();
  std::array<SourceClaim, 8> claims;
  std::array<Result, 8> results{};
  std::array<std::thread, 8> threads;
  for (size_t i = 0; i < threads.size(); ++i)
    threads[i] = std::thread([&, i] {
      results[i] = domain.reserve(ResourceKind::Output, claims[i]);
    });
  for (auto &thread : threads) thread.join();
  uint32_t accepted = 0;
  for (size_t i = 0; i < claims.size(); ++i) {
    if (results[i] == Result::Ok) ++accepted;
    else require(results[i] == Result::WouldBlock && !claims[i], "bad concurrent refusal");
  }
  require(accepted == 2 && snapshot(domain).payloadBytes == 720,
          "concurrent output admission exceeded atomic role/byte cap");
  for (auto &claim : claims)
    if (claim) ok(claim.cancelUnaccepted(), "known preacceptance rollback");
  require(snapshot(domain).payloadBytes == 0, "known rollback did not refund");
  close(root);
}
void creationIdentityIsolation() {
  int gpu = 0, device = 0, queue = 0, instance = 0, other = 0;
  const mgpu::RetainedCreationIdentity legacy{
      &gpu, &device, &queue, &instance, 11, 22, {}};
  auto legacyQueue = legacy;
  legacyQueue.queue = &other;
  require(mgpu::samePrivateCreation(legacy, legacyQueue) &&
              !mgpu::sameRetainedCreation(legacy, legacyQueue) &&
              mgpu::sameQuarantinedCreation(legacy, legacyQueue),
          "legacy comparison behavior changed");
  auto first = create();
  auto second = create();
  auto source = legacy;
  source.source = first->domain();
  auto foreign = legacy;
  foreign.source = second->domain();
  require(mgpu::samePrivateCreation(source, source) &&
              mgpu::sameRetainedCreation(source, source) &&
              mgpu::sameQuarantinedCreation(source, source),
          "same retained source identity refused");
  require(!mgpu::samePrivateCreation(source, foreign) &&
              !mgpu::sameRetainedCreation(source, foreign) &&
              !mgpu::sameQuarantinedCreation(source, foreign),
          "matching devices/generations collapsed distinct source roots");
  require(!mgpu::samePrivateCreation(source, legacy) &&
              !mgpu::samePrivateCreation(legacy, source) &&
              !mgpu::sameQuarantinedCreation(source, legacy) &&
              !mgpu::sameQuarantinedCreation(legacy, source),
          "scoped/unscoped resource or quarantine mixed");
  auto changed = source;
  changed.queue = &other;
  require(!mgpu::samePrivateCreation(source, changed), "scoped queue mismatch");
  changed = source;
  changed.instance = &other;
  require(!mgpu::samePrivateCreation(source, changed), "scoped instance mismatch");
  changed = source;
  changed.deviceGeneration++;
  require(!mgpu::samePrivateCreation(source, changed), "scoped generation mismatch");
  changed = source;
  changed.queue = nullptr;
  require(!mgpu::samePrivateCreation(changed, changed), "incomplete scoped creation accepted");
  close(first);
  close(second);
}
void metadataAllocationLifetime() {
  using mgpu::capture_cleanup::Family;
  namespace layout = mgpu::allocation_layout;
#ifdef MINIGPU_EXPECT_KNOWN_ALLOCATION_LAYOUT
  static_assert(layout::sharedLayoutSupported, "Known-layout witness needs the qualified Release compiler/STL/configuration");
#endif
  const std::array<layout::Expected, 2> expected{{operationStateAllocationExtent(),
      mgpu::capture_cleanup::familyStateAllocationExtent()}};
  const auto build = layout::identity();
  require(build.version == 1 && build.pointerBytes == sizeof(void *) &&
      build.defaultNewAlignment == __STDCPP_DEFAULT_NEW_ALIGNMENT__, "compiled layout identity");
  for (const auto &entry : expected) {
    require((entry.status == layout::Status::Known) == layout::sharedLayoutSupported,
        "unsupported layout cannot report a known extent");
    require(layout::sharedLayoutSupported ? entry.extent.bytes && entry.extent.alignment :
        entry.extent.bytes == 0 && entry.extent.alignment == 0, "explicit allocation layout status");
  }
  auto root = create();
  auto domain = root->domain();
  uint32_t reentrant = 0;
  metadataAllocateHook = [&](MetadataKind kind) {
    if (kind != MetadataKind::OperationState) return;
    Family refused;
    require(Family::beginWork(domain, refused) == Result::WouldBlock && !refused,
            "reentrant family reached an allocator before operation admission");
    ++reentrant;
  };
  Family family;
  ok(Family::beginWork(domain, family), "funded family allocation");
  metadataAllocateHook = {};
  for (size_t i = 0; i < expected.size(); ++i) {
    if (expected[i].status == layout::Status::Known)
      require(allocationBytes[i].load() == expected[i].extent.bytes &&
          allocationAlignments[i].load() == expected[i].extent.alignment,
          "actual production operation/family control-block request differs from compiled extent");
  }
  std::cout << "allocation layout " << (layout::sharedLayoutSupported ? "Known" : "Unsupported")
      << " compiler=" << build.compilerFullVersion << " stl=" << build.stlUpdate
      << " language=" << build.languageVersion << " flags=" << build.buildFlags
      << " pointer=" << build.pointerBytes << " new-alignment=" << build.defaultNewAlignment
      << " operation=" << allocationBytes[0].load() << '/' << allocationAlignments[0].load()
      << " family=" << allocationBytes[1].load() << '/' << allocationAlignments[1].load() << '\n';
  require(reentrant == 1 && snapshot(domain).metadataAllocations ==
      std::array<uint32_t, metadataKinds>{1, 1}, "covered allocation inventory");
  const auto opAllocations = allocationAttempts[0].load();
  const auto familyAllocations = allocationAttempts[1].load();
  std::array<Result, 8> results{};
  std::array<std::thread, 8> threads;
  for (size_t i = 0; i < threads.size(); ++i) threads[i] = std::thread([&, i] {
    Family refused;
    results[i] = Family::beginWork(domain, refused);
  });
  for (auto &thread : threads) thread.join();
  for (auto result : results) require(result == Result::WouldBlock, "concurrent family refusal");
  require(allocationAttempts[0].load() == opAllocations &&
      allocationAttempts[1].load() == familyAllocations,
      "occupied concurrent attempts allocated metadata");
  auto weak = family.allocationWeakForTest();
  family = {};
  require(weak.expired() && snapshot(domain).operationFamilies == 1 &&
      snapshot(domain).metadataAllocations == std::array<uint32_t, metadataKinds>{1, 1},
      "payload destruction refunded the still-retained shared control block");
  ok(root->seal(), "seal retained allocation");
  JoinReceipt joined;
  require(root->tryJoin(joined) == Result::WouldBlock, "join passed weak allocation tail");
  uint32_t deallocations = 0;
  bool blocked = true;
  metadataDeallocateHook = [&](MetadataKind kind) {
    const auto held = snapshot(domain);
    Family refused;
    JoinReceipt receipt;
    blocked = blocked && held.metadataAllocations[static_cast<size_t>(kind)] == 1 &&
        held.operationFamilies == 1 && root->tryJoin(receipt) == Result::WouldBlock &&
        Family::begin(domain, refused) == Result::WouldBlock;
    ++deallocations;
  };
  weak.reset();
  metadataDeallocateHook = {};
  for (size_t i = 0; i < expected.size(); ++i)
    if (expected[i].status == layout::Status::Known)
      require(allocationBytes[i].load() == expected[i].extent.bytes &&
          allocationAlignments[i].load() == expected[i].extent.alignment,
          "retained weak/free tail changed its exact allocation extent");
  require(blocked && deallocations == 2 && snapshot(domain).operationFamilies == 0 &&
      snapshot(domain).metadataAllocations == std::array<uint32_t, metadataKinds>{0, 0},
      "deallocation itself did not retain exact operation/family slots");
  close(root);

  for (auto kind : {MetadataKind::OperationState, MetadataKind::FamilyState}) {
    root = create(); domain = root->domain();
    failMetadataKind = static_cast<int>(kind);
    Family refused;
    require(Family::beginWork(domain, refused) == Result::Internal && !refused,
            "injected allocation failure did not refuse");
    require(snapshot(domain).operationFamilies == 0 && snapshot(domain).phase == Phase::Open &&
        snapshot(domain).metadataAllocations == std::array<uint32_t, metadataKinds>{0, 0},
        "known allocation rollback leaked or poisoned covered capacity");
    ok(Family::beginWork(domain, family), "reuse after known metadata allocation failure");
    family = {};
    close(root);
  }
  root = create(); domain = root->domain();
  SourceOperation operation;
  ok(domain.beginOperation(OperationPurpose::Work, operation), "constructor failure operation");
  auto view = operation.view();
  MetadataAllocation allocation;
  ok(view.reserveFamilyMetadata(allocation), "constructor failure slot");
  struct Throwing { Throwing() { throw std::runtime_error("constructor refused"); } };
  bool constructorThrew = false;
  try {
    (void)std::allocate_shared<Throwing>(MetadataAllocator<Throwing>(allocation, view));
  } catch (const std::runtime_error &) { constructorThrew = true; }
  require(constructorThrew, "injected constructor did not throw");
  require(snapshot(domain).metadataAllocations == std::array<uint32_t, metadataKinds>{1, 0} &&
      snapshot(domain).phase == Phase::Open, "constructor failure was not a known rollback");
  ok(operation.finish(), "finish constructor failure operation");
  view = {};
  close(root);
}
void metadataLayoutVariantRefusals() {
  namespace layout = mgpu::allocation_layout;
  if (!layout::sharedLayoutSupported) return; // Known mode is explicitly required by its focused CMake option.
  // Synthetic extents isolate schema validation. Actual 34 private types and
  // their compiled extents are checked by the exported native cold-query case.
  const auto fill = [](layout::MetadataLayoutBuilder &builder, MetadataKind omittedKind,
                       uint32_t omittedVariant, bool wrongOrdinaryVariant) {
    for (size_t i = 0; i < metadataKinds; ++i) {
      const auto kind = static_cast<MetadataKind>(i);
      if (kind == MetadataKind::PrimaryJob || kind == MetadataKind::PrimaryAnchor) {
        const uint32_t last = kind == MetadataKind::PrimaryJob ? MGPU_CAPTURE_META_VARIANT_PRIVATE :
            MGPU_CAPTURE_META_VARIANT_CLEANUP;
        for (uint32_t variant = MGPU_CAPTURE_META_VARIANT_WORK; variant <= last; ++variant)
          if (kind != omittedKind || variant != omittedVariant) builder.unique<uint64_t>(kind, variant);
      } else builder.unique<uint64_t>(kind, wrongOrdinaryVariant && kind == MetadataKind::OperationState ?
          MGPU_CAPTURE_META_VARIANT_WORK : MGPU_CAPTURE_META_VARIANT_GENERIC);
    }
  };
  MGPUCaptureMetadataLayoutV1 out{};
  layout::MetadataLayoutBuilder complete, missingJob, missingAnchor, wrongKind;
  fill(complete, MetadataKind::Count, 0, false);
  fill(missingJob, MetadataKind::PrimaryJob, MGPU_CAPTURE_META_VARIANT_PRIVATE, false);
  fill(missingAnchor, MetadataKind::PrimaryAnchor, MGPU_CAPTURE_META_VARIANT_CLEANUP, false);
  fill(wrongKind, MetadataKind::Count, 0, true);
  require(complete.finish(out) && out.slot_count == 60 &&
      !missingJob.finish(out) && !missingAnchor.finish(out) && !wrongKind.finish(out),
      "partial or misplaced variants cannot publish a complete layout");
}
[[maybe_unused]] void boundMetadataAllocationLifetime(const MGPUCaptureMetadataLayoutV1 &layout) {
  using mgpu::capture_cleanup::Family;
  const auto opBytes = layout.kinds[MGPU_CAPTURE_META_OPERATION_STATE].extents[0].bytes;
  const auto familyBytes = layout.kinds[MGPU_CAPTURE_META_FAMILY_STATE].extents[0].bytes;
  const auto familyAlignment = layout.kinds[MGPU_CAPTURE_META_FAMILY_STATE].extents[0].alignment;
  const auto heldBytes = opBytes + familyBytes;
  auto root = create();
  for (unsigned fault = 0; fault < 9; ++fault) {
    auto bad = layout;
    switch (fault) {
    case 0: --bad.struct_size; break;
    case 1: ++bad.compiler_full_version; break;
    case 2: --bad.total_requested_bytes; break;
    case 3: bad.flags |= 2; break;
    case 4: bad.reserved = 1; break;
    case 5: bad.kinds[0].extents[0].alignment = 3; break;
    case 6: bad.kinds[0].extents[2].bytes = 8; break;
    case 7: bad.kinds[MGPU_CAPTURE_META_PRIMARY_JOB].extent_count = 2; break;
    case 8: bad.kinds[MGPU_CAPTURE_META_RESOURCE_STATE].extents[0].bytes = UINT64_MAX - 7; break;
    }
    require(root->bindMetadataLayout(bad) == Result::Invalid &&
        !snapshot(root->domain()).metadataLayoutBound, "malformed/build-incompatible table changed the root");
  }
  auto caller = layout;
  ok(root->bindMetadataLayout(caller), "bind original compiled layout");
  require(root->bindMetadataLayout(layout) == Result::Invalid, "duplicate bind changed immutable admission");
  caller.total_requested_bytes = 8; caller.kinds[0].extents[0].bytes = 8;
  const auto domain = root->domain();
  Family family;
  uint32_t reentrant = 0;
  metadataAllocateHook = [&](MetadataKind kind) {
    if (kind == MetadataKind::OperationState) {
      Family refused;
      require(snapshot(domain).metadataAllocationBytes == opBytes &&
          Family::beginWork(domain, refused) == Result::WouldBlock,
          "bytes were not charged before heap/reentrant entry");
      ++reentrant;
    }
  };
  ok(Family::beginWork(domain, family), "actual family against compiled bound table");
  metadataAllocateHook = {};
  auto held = snapshot(domain);
  require(reentrant == 1 && held.metadataLayoutBound && held.metadataAllocationLimit == layout.total_requested_bytes &&
      held.metadataAllocationBytes == heldBytes && held.metadataBytesByKind[0] == opBytes &&
      held.metadataBytesByKind[1] == familyBytes, "immutable copy/per-kind and total request charges");
  std::array<Result, 2> results{};
  std::array<std::thread, 2> threads;
  for (size_t i = 0; i < threads.size(); ++i) threads[i] = std::thread([&, i] {
    Family refused; results[i] = Family::beginWork(domain, refused);
  });
  for (auto &thread : threads) thread.join();
  for (auto result : results) require(result == Result::WouldBlock, "bound concurrent admission borrowed a family");
  auto weak = family.allocationWeakForTest();
  family = {};
  require(weak.expired() && snapshot(domain).metadataAllocationBytes == heldBytes,
      "payload destruction refunded weak-control-block bytes");
  ok(root->seal(), "seal bound retained family");
  uint32_t frees = 0;
  metadataDeallocateHook = [&](MetadataKind kind) {
    const auto current = snapshot(domain);
    JoinReceipt joined;
    require(current.metadataAllocationBytes == (kind == MetadataKind::FamilyState ? heldBytes : opBytes) &&
        current.metadataBytesByKind[static_cast<size_t>(kind)] ==
            (kind == MetadataKind::FamilyState ? familyBytes : opBytes) &&
        root->tryJoin(joined) == Result::WouldBlock, "deallocation hook ran after byte refund or join");
    ++frees;
  };
  weak.reset(); metadataDeallocateHook = {};
  require(frees == 2 && snapshot(domain).metadataAllocationBytes == 0 &&
      snapshot(domain).metadataBytesByKind == std::array<uint64_t, metadataKinds>{},
      "known actual final frees did not refund exact bytes");
  close(root);

  root = create();
  SourceClaim pin;
  ok(root->domain().reserve(ResourceKind::ScopedPin, pin), "claim before attempted late bind");
  ok(pin.cancelUnaccepted(), "cancel prebind claim");
  require(root->bindMetadataLayout(layout) == Result::Invalid &&
      snapshot(root->domain()).resources == std::array<uint32_t, resourceKinds>{},
      "zero counters hid a previous resource admission");
  close(root);
  root = create(); failMetadataKind = static_cast<int>(MetadataKind::OperationState);
  Family refused;
  require(Family::beginWork(root->domain(), refused) == Result::Internal &&
      root->bindMetadataLayout(layout) == Result::Invalid,
      "known failed operation allowed retroactive layout binding");
  close(root);

  root = create(); ok(root->bindMetadataLayout(layout), "bind mismatch case");
  auto mismatchDomain = root->domain();
  SourceOperation operation;
  ok(mismatchDomain.beginOperation(OperationPurpose::Work, operation), "mismatch operation");
  auto view = operation.view();
  MetadataAllocation allocation;
  ok(view.reserveFamilyMetadata(allocation), "mismatch reserved identity");
  const auto attempts = allocationAttempts[1].load();
  for (bool wrongSize : {false, true}) {
    bool rejected = false;
    try {
      (void)allocation.allocateBytes(static_cast<size_t>(familyBytes) + (wrongSize ? familyAlignment : 0),
          wrongSize ? familyAlignment : familyAlignment == 1 ? 2 : 1);
    } catch (const MetadataAllocationError &error) { rejected = error.result == Result::Invalid; }
    require(rejected && allocationAttempts[1].load() == attempts &&
        snapshot(mismatchDomain).metadataAllocationBytes == opBytes &&
        snapshot(mismatchDomain).metadataBytesByKind[1] == 0,
        "extent mismatch reached heap hook or charged bytes");
  }
  ok(allocation.cancelUnallocated(), "cancel mismatched extent reservation");
  ok(operation.finish(), "finish mismatch operation"); view = {};
  require(snapshot(mismatchDomain).metadataAllocationBytes == 0, "mismatch left byte debt");
  close(root);

  for (auto kind : {MetadataKind::OperationState, MetadataKind::FamilyState}) {
    root = create(); ok(root->bindMetadataLayout(layout), "bind fault case");
    failMetadataKind = static_cast<int>(kind);
    require(Family::beginWork(root->domain(), refused) == Result::Internal &&
        snapshot(root->domain()).metadataAllocationBytes == 0 &&
        snapshot(root->domain()).metadataBytesByKind == std::array<uint64_t, metadataKinds>{} &&
        snapshot(root->domain()).phase == Phase::Open, "known allocation failure did not refund bytes");
    ok(Family::beginWork(root->domain(), family), "bound retry after known allocation failure");
    family = {};
    require(snapshot(root->domain()).metadataAllocationBytes == 0, "retry free retained request bytes");
    close(root);
  }
  root = create(); ok(root->bindMetadataLayout(layout), "bind known-free/other-poison case");
  ok(Family::beginWork(root->domain(), family), "known family before independent poison");
  root->domain().quarantine(); family = {};
  require(snapshot(root->domain()).phase == Phase::Poisoned && snapshot(root->domain()).metadataAllocationBytes == 0,
      "known free on poisoned root retained unrelated byte charge");
  root.reset(); // Poison is not join/refund authority.

  root = create(); ok(root->bindMetadataLayout(layout), "bind unknown case");
  const auto unknown = root->domain();
  ok(Family::beginWork(unknown, family), "actual family before unknown disposal");
  family.quarantine(); family = {};
  require(snapshot(unknown).phase == Phase::Poisoned && snapshot(unknown).metadataAllocationBytes == heldBytes &&
      snapshot(unknown).metadataBytesByKind[0] == opBytes && snapshot(unknown).metadataBytesByKind[1] == familyBytes,
      "physical frees of uncertain actual family/operation erased admitted debt");
  JoinReceipt joined;
  require(root->tryJoin(joined) == Result::Poisoned && !joined, "uncertain byte debt supplied a joined receipt");
  root.reset();
  std::cout << "bound metadata: actual DLL layout, pre-new extent checks, exact bytes, weak/free/fault/unknown tails PASS\n";
}
void boundMetadataFromOriginalDll(const char *path) {
#ifdef _WIN32
  require(mgpu::allocation_layout::sharedLayoutSupported, "bound CPU witness requires qualified Known configuration");
  const std::filesystem::path absolute(path);
  require(absolute.is_absolute(), "supply explicit absolute scratch DLL path");
  struct Module {
    HMODULE value;
    ~Module() { if (value) FreeLibrary(value); }
  } module{LoadLibraryW(absolute.c_str())};
  require(module.value != nullptr, "load original cold layout DLL and its installed dependencies");
  using Query = int (*)(MGPUCaptureMetadataLayoutV1 *, uint32_t);
  Query query = nullptr;
  auto address = GetProcAddress(module.value, "mgpuCaptureSourceGetMetadataLayoutV1");
  static_assert(sizeof(address) == sizeof(query));
  std::memcpy(&query, &address, sizeof(query));
  require(query != nullptr, "original DLL has layout export");
  MGPUCaptureMetadataLayoutV1 layout{};
  require(query(&layout, sizeof(layout)) == 0 && mgpu::allocation_layout::validateMetadataLayout(layout),
      "original DLL supplied compatible complete compiled layout");
  using QueryCpu = int (*)(MGPUCaptureSourceCpuMetadataProfileV1 *, uint32_t);
  QueryCpu queryCpu = nullptr;
  address = GetProcAddress(module.value, "mgpuCaptureSourceGetCpuMetadataProfileV1");
  static_assert(sizeof(address) == sizeof(queryCpu));
  std::memcpy(&queryCpu, &address, sizeof(queryCpu));
  require(queryCpu != nullptr, "original DLL has additive CPU metadata attestation");
  MGPUCaptureSourceCpuMetadataProfileV1 profile{};
  require(queryCpu(&profile, sizeof(profile)) == 0 &&
      mgpu::allocation_layout::validateCpuMetadataProfile(profile) &&
      std::memcmp(&profile.layout, &layout, sizeof(layout)) == 0,
      "attested enforced table matches the original DLL's exact reporting table");
  for (unsigned field = 0; field < 7; ++field) {
    auto bad = profile;
    switch (field) {
    case 0: --bad.struct_size; break;
    case 1: ++bad.version; break;
    case 2: ++bad.enforcement_revision; break;
    case 3: bad.flags |= 2; break;
    case 4: ++bad.logical.coverage; break;
    case 5: ++bad.layout.total_requested_bytes; break;
    case 6: ++bad.reserved; break;
    }
    require(!mgpu::allocation_layout::validateCpuMetadataProfile(bad), "malformed CPU attestation accepted");
  }
  boundMetadataAllocationLifetime(profile.layout);
  std::cout << "attested metadata: original DLL profile validated and bound before production allocator entry PASS\n";
#else
  (void)path;
  throw std::runtime_error("bound native layout witness unsupported on this platform");
#endif
}
void operationAndNativeTails() {
  auto root = create();
  auto domain = root->domain();
  SourceOperation operation, competing;
  ok(domain.beginOperation(OperationPurpose::Work, operation), "begin work");
  auto view = operation.view();
  SourceClaim primary, fence, wait, ticket, duplicate;
  ok(view.reserveContinuation(ContinuationKind::PrimaryTask, primary), "primary");
  ok(primary.accept(), "accept primary");
  ok(view.reserveCompletionTicket(ticket), "ticket");
  ok(ticket.accept(), "accept ticket");
  require(domain.beginOperation(OperationPurpose::Work, competing) == Result::WouldBlock &&
              !competing, "public/reentrant operation borrowed existing family");
  require(view.reserveContinuation(ContinuationKind::PrimaryTask, duplicate) ==
              Result::WouldBlock &&
              view.reserveCompletionTicket(duplicate) == Result::WouldBlock,
          "duplicate primary or ticket exceeded family cap");
  SourceOperation moved(std::move(operation));
  require(!operation && moved, "operation move did not transfer custody");
  ok(root->seal(), "seal while work accepted");
  require(domain.beginOperation(OperationPurpose::Work, competing) == Result::Sealed,
          "seal admitted new work");
  ok(view.reserveContinuation(ContinuationKind::QueueFence, fence), "nested fence after seal");
  ok(fence.accept(), "accept fence after seal");
  ok(view.reserveContinuation(ContinuationKind::QueueWait, wait), "nested wait after seal");
  ok(wait.accept(), "accept wait after seal");
  require(view.reserveContinuation(ContinuationKind::Count, duplicate) == Result::Invalid,
          "invalid continuation");
  ok(moved.finish(), "parent callback returned");
  view = {};
  auto publicTicket = std::make_shared<SourceClaim>(std::move(ticket));
  auto nativeTicketRef = publicTicket;
  publicTicket.reset();
  auto held = snapshot(domain);
  require(held.operationFamilies == 1 && held.completionTickets == 1 &&
              held.continuations == std::array<uint32_t, continuationKinds>{1, 1, 1},
          "callback/public ticket release refunded native tails");
  JoinReceipt joined;
  require(root->tryJoin(joined) == Result::WouldBlock, "joined native tails");
  ok(primary.retire(), "last primary task capture destroyed");
  ok(fence.retire(), "last fence capture destroyed");
  ok(wait.retire(), "last queue wait destroyed");
  require(domain.beginOperation(OperationPurpose::Cleanup, competing) == Result::WouldBlock,
          "cleanup reused family before ticket NativeRef ended");
  ok(nativeTicketRef->retire(), "final ticket NativeRef disposition");
  nativeTicketRef.reset();
  require(snapshot(domain).operationFamilies == 0, "last tail did not release family");
  ok(domain.beginOperation(OperationPurpose::Cleanup, competing), "sealed cleanup operation");
  auto cleanupView = competing.view();
  ok(competing.finish(), "cleanup callback returned");
  require(root->tryJoin(joined) == Result::WouldBlock,
          "retained cleanup task view was omitted from join");
  cleanupView = {};
  ok(root->tryJoin(joined), "join final operation tail");
  ok(root->release(joined), "release after operation join");
  root.reset();
}
void uncertainDispositionsNeverRefund() {
  const auto before = registeredSources();
  auto root = create();
  auto domain = root->domain();
  {
    SourceClaim abandoned;
    ok(domain.reserve(ResourceKind::Output, abandoned), "reserve abandoned output");
    ok(abandoned.accept(), "accept abandoned output");
  }
  auto held = snapshot(domain);
  require(held.phase == Phase::Poisoned && held.payloadBytes == 360 &&
              held.resources[static_cast<size_t>(ResourceKind::Output)] == 1,
          "uncertain output refunded instead of poisoning");
  JoinReceipt refused;
  require(root->seal() == Result::Poisoned &&
              root->tryJoin(refused) == Result::Poisoned,
          "poisoned source joined");
  root.reset();

  root = create();
  domain = root->domain();
  {
    SourceOperation abandoned;
    ok(domain.beginOperation(OperationPurpose::Work, abandoned), "abandoned operation");
  }
  require(snapshot(domain).phase == Phase::Poisoned &&
              snapshot(domain).operationFamilies == 1 &&
              snapshot(domain).metadataAllocations[0] == 1,
          "uncertain operation family refunded");
  SourceOperation cleanup;
  require(domain.beginOperation(OperationPurpose::Cleanup, cleanup) == Result::WouldBlock,
          "unknown family was reused for cleanup");
  root.reset();

  root = create();
  domain = root->domain();
  root.reset();
  require(snapshot(domain).phase == Phase::Poisoned && registeredSources() == before + 3,
          "abandoned controls silently freed process roots");

  root = create();
  domain = root->domain();
  SourceClaim pin;
  ok(domain.reserve(ResourceKind::ScopedPin, pin), "reserve quarantined pin");
  ok(pin.accept(), "accept quarantined pin");
  domain.quarantine();
  ok(pin.retire(), "known pin disposition after unrelated uncertainty");
  require(snapshot(domain).phase == Phase::Poisoned &&
              root->seal() == Result::Poisoned &&
              root->tryJoin(refused) == Result::Poisoned,
          "native quarantine became joined refund authority");
  root.reset();
  require(registeredSources() == before + 4, "native quarantine lost process root");
}
void boundedRegistry() {
  const auto retained = registeredSources();
  std::vector<SourceRootOwner> roots;
  for (uint32_t i = retained; i < maxSources; ++i) roots.push_back(create());
  SourceRootOwner refused;
  require(registeredSources() == maxSources &&
              SourceRoot::create({10, 9, 1, 0}, refused) == Result::WouldBlock && !refused,
          "process source registry is not bounded");
  const auto oldIdentity = snapshot(roots.back()->domain()).identity;
  close(roots.back());
  roots.back() = create();
  require(snapshot(roots.back()->domain()).identity > oldIdentity,
          "freed registry slot recycled source identity");
  for (auto &root : roots) close(root);
  require(registeredSources() == retained, "explicit joined cleanup changed poisoned roots");
}
} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 1 || (argc == 3 && std::strcmp(argv[1], "--layout-dll") == 0),
        "usage: minigpu_source_scope_cpu [--layout-dll ABSOLUTE_DLL_PATH]");
    concurrentColdRegistry();
    geometryAndInventory();
    domainsAliasesAndAtomicCapacity();
    creationIdentityIsolation();
    operationAndNativeTails();
    metadataAllocationLifetime();
    metadataLayoutVariantRefusals();
    if (argc == 3) boundMetadataFromOriginalDll(argv[2]);
    uncertainDispositionsNeverRefund();
    boundedRegistry();
    std::cout << "PASS source scope: fixed inventory, exact domains, atomic capacity, "
                 "native tails, sealed cleanup, poison and bounded registry\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL source scope: " << error.what() << '\n';
    return 1;
  }
}

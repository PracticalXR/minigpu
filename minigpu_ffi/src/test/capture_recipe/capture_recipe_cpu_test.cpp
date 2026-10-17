#include "../../include/minigpu_capture_recipe_owner.h"
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
using namespace mgpu::source_scope;
namespace recipe = mgpu::capture_recipe;
namespace fixed = mgpu::capture_recipe_v1;
void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
void ok(Result value, const char *message) { require(value == Result::Ok, message); }
SourceRootOwner create() {
  SourceRootOwner root;
  ok(SourceRoot::create({10, 9, 1, 0}, root), "create");
  return root;
}
Snapshot snapshot(const SourceDomain &source) {
  Snapshot value;
  ok(source.snapshot(value), "snapshot");
  return value;
}
void close(SourceRootOwner &root) {
  ok(root->seal(), "seal");
  JoinReceipt joined;
  ok(root->tryJoin(joined), "join");
  ok(root->release(joined), "release");
  root.reset();
}
struct Probe {
  SourceDomain source;
  ResourceKind role;
  bool &intact;
  ~Probe() {
    Snapshot value;
    const auto result = source.snapshot(value);
    intact = intact && result == Result::Ok &&
        (role == ResourceKind::Count ?
            value.operationFamilies == 1 && value.continuations[0] == 1 :
            value.resources[static_cast<size_t>(role)] == 1);
  }
};
struct State : recipe::ResourceCustody {
  State(const SourceDomain &source, ResourceKind role, bool &intact)
      : probe{source, role, intact} { ok(reserveSourceRole(source, role), "role reservation"); }
  Probe probe;
  std::shared_ptr<State> dependency;
};
struct Job : recipe::JobCustody {
  Job(const SourceDomain &source, bool &intact)
      : probe{source, ResourceKind::Count, intact} {
    ok(prepareSourceJob(source), "prepare job");
  }
  Probe probe;
  std::string payload = "owned payload retained by caller and worker";
};
SourceDomain refundSource;
int refunds = 0;
bool failRefund = false;
bool refundGlobal() noexcept {
  Snapshot value;
  if (refundSource.snapshot(value) != Result::Ok || value.operationFamilies != 1)
    return false;
  if (failRefund) return false;
  ++refunds;
  return true;
}
void exactRecipeValidation() {
  auto root = create(), foreign = create();
  const auto source = root->domain();
  ResourceKind role;
  ok(recipe::classifyResource(source, 0, 0, nullptr, 0, role), "shader descriptor");
  require(role == ResourceKind::RecipeShader, "shader role");
  ok(recipe::classifyResource(source, 360, fixed::packedUsage, nullptr, 0, role), "packed descriptor");
  require(role == ResourceKind::PackedBuffer, "packed role");
  const std::array<uint8_t, 16> dimensions{10, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  ok(recipe::classifyResource(source, 16, fixed::parametersUsage, dimensions.data(), 16, role), "dimensions descriptor");
  require(role == ResourceKind::Parameters, "dimensions role");
  for (auto size : {356ull, 364ull})
    require(recipe::classifyResource(source, size, fixed::packedUsage, nullptr, 0, role) == Result::Invalid,
            "resized packed buffer admitted");
  require(recipe::classifyResource(source, 360, 0x8c, nullptr, 0, role) == Result::Invalid,
          "extra usage admitted");
  require(recipe::classifyResource(source, 360, fixed::packedUsage, dimensions.data(), 16, role) == Result::Invalid,
          "packed initialization admitted");
  auto bad = dimensions;
  bad[8] = 1;
  require(recipe::classifyResource(source, 16, fixed::parametersUsage, bad.data(), 16, role) == Result::Invalid,
          "nonzero parameter padding admitted");
  bad = dimensions;
  bad[0] = 9;
  require(recipe::classifyResource(source, 16, fixed::parametersUsage, bad.data(), 16, role) == Result::Invalid,
          "different geometry admitted");
  std::array<recipe::Binding, 2> bindings{{
      {fixed::packedSlot, 1, source, ResourceKind::PackedBuffer, 0, 360},
      {fixed::parametersSlot, 3, source, ResourceKind::Parameters, 0, 16}}};
  const auto valid = [&] {
    return recipe::validateSetup(source, source, ResourceKind::RecipeShader,
        fixed::wgsl.data(), fixed::wgsl.size(), fixed::frameSlot, bindings);
  };
  ok(valid(), "fixed setup");
  std::swap(bindings[0], bindings[1]);
  ok(valid(), "binding order is not new authority");
  std::swap(bindings[0], bindings[1]);
  const auto original = bindings;
  bindings[0].source = foreign->domain();
  require(valid() == Result::Invalid, "foreign source binding admitted");
  bindings = original; bindings[0].role = ResourceKind::Parameters;
  require(valid() == Result::Invalid, "wrong buffer role admitted");
  bindings = original; bindings[0].offset = 4;
  require(valid() == Result::Invalid, "offset binding admitted");
  bindings = original; bindings[0].bytes = 356;
  require(valid() == Result::Invalid, "partial packed range admitted");
  bindings = original; bindings[1] = bindings[0];
  require(valid() == Result::Invalid, "duplicate slot admitted");
  bindings = original;
  auto modified = std::string(fixed::wgsl); modified[1] = 'x';
  require(recipe::validateSetup(source, source, ResourceKind::RecipeShader,
      modified.data(), modified.size(), 0, bindings) == Result::Invalid, "changed WGSL admitted");
  require(recipe::validateSetup(source, foreign->domain(), ResourceKind::RecipeShader,
      fixed::wgsl.data(), fixed::wgsl.size(), 0, bindings) == Result::Invalid, "foreign shader admitted");
  ok(root->seal(), "seal recipe source");
  require(valid() == Result::Sealed && recipe::workStatus(source) == Result::Sealed,
          "sealed work still admitted");
  close(root); close(foreign);
}
void rolesSurviveAliasesAndMemberDestruction() {
  auto root = create();
  const auto source = root->domain();
  bool intact = true;
  auto shader = std::make_shared<State>(source, ResourceKind::RecipeShader, intact);
  auto packed = std::make_shared<State>(source, ResourceKind::PackedBuffer, intact);
  ok(shader->acceptSourceRole(), "accept shader");
  ok(packed->acceptSourceRole(), "accept packed");
  shader->dependency = packed;
  packed->proveNativeReleased();
  packed.reset(); // Native buffer-wrapper release is not last binding-State release.
  recipe::ResourceCustody refused;
  require(refused.reserveSourceRole(source, ResourceKind::PackedBuffer) == Result::WouldBlock,
          "shader alias allowed a second backing envelope");
  shader->proveNativeReleased();
  auto tail = shader;
  shader.reset();
  require(snapshot(source).payloadBytes == 360, "public shader release refunded retained aliases");
  tail.reset();
  require(intact && snapshot(source).payloadBytes == 0,
          "source role refunded before derived members or did not retire");
  {
    recipe::ResourceCustody retry;
    ok(retry.reserveSourceRole(source, ResourceKind::PackedBuffer), "role retry");
    // Known preacceptance rollback needs no invented native release proof.
  }
  require(snapshot(source).payloadBytes == 0, "unaccepted role rollback failed");
  close(root);
}
void finalJobTailsAndKnownRejection() {
  auto root = create();
  const auto source = root->domain();
  refundSource = source; refunds = 0;
  bool intact = true;
  auto caller = std::make_shared<Job>(source, intact);
  caller->setGlobalJobRefund(refundGlobal);
  ok(caller->acceptSourceJob(), "accept primary task");
  std::function<void()> workerTail = [caller] { caller->proveJobFinished(); };
  workerTail(); // Promise/callback has completed; its captured owner still exists.
  caller.reset();
  recipe::JobCustody next;
  require(next.prepareSourceJob(source) == Result::WouldBlock,
          "new operation borrowed an old worker tail");
  require(refunds == 0 && snapshot(source).operationFamilies == 1,
          "promise completion refunded job");
  workerTail = {};
  require(intact && refunds == 1 && snapshot(source).operationFamilies == 0,
          "final member/task destruction did not dispose job");
  {
    Job rejected(source, intact);
    rejected.setGlobalJobRefund(refundGlobal);
    ok(rejected.acceptSourceJob(), "accepted enqueue attempt");
    rejected.proveEnqueueRejected();
  }
  require(refunds == 2 && snapshot(source).operationFamilies == 0,
          "known enqueue refusal retained source family");
  try {
    Job payloadFailure(source, intact);
    payloadFailure.setGlobalJobRefund(refundGlobal);
    throw std::bad_alloc();
  } catch (const std::bad_alloc &) {}
  require(intact && refunds == 3 && snapshot(source).operationFamilies == 0,
          "preacceptance allocation failure did not roll back");
  close(root);
}
void sealBeforeAcceptance() {
  auto root = create();
  const auto source = root->domain();
  {
    recipe::ResourceCustody resource;
    recipe::JobCustody job;
    ok(resource.reserveSourceRole(source, ResourceKind::PackedBuffer), "reserved before seal");
    ok(job.prepareSourceJob(source), "job before seal");
    ok(root->seal(), "seal before native entry");
    require(resource.acceptSourceRole() == Result::Sealed &&
                job.acceptSourceJob() == Result::Sealed, "late native entry admitted");
  }
  require(snapshot(source).payloadBytes == 0 && snapshot(source).operationFamilies == 0,
          "sealed known rollback retained live claims");
  close(root);
}
void unknownWorkRetainsAccounting() {
  auto root = create();
  auto source = root->domain();
  {
    recipe::ResourceCustody missingRelease;
    ok(missingRelease.reserveSourceRole(source, ResourceKind::PackedBuffer), "unknown resource");
    ok(missingRelease.acceptSourceRole(), "accepted resource");
  }
  require(snapshot(source).phase == Phase::Poisoned && snapshot(source).payloadBytes == 360,
          "missing native release became refund");
  root.reset();
  root = create(); source = root->domain(); refundSource = source;
  const auto before = refunds;
  {
    recipe::JobCustody missingTail;
    ok(missingTail.prepareSourceJob(source), "unknown task");
    missingTail.setGlobalJobRefund(refundGlobal);
    ok(missingTail.acceptSourceJob(), "accepted unknown task");
  }
  require(refunds == before && snapshot(source).phase == Phase::Poisoned &&
              snapshot(source).operationFamilies == 1, "unknown job refunded");
  root.reset();
  root = create(); source = root->domain(); refundSource = source; failRefund = true;
  {
    recipe::JobCustody failedRefund;
    ok(failedRefund.prepareSourceJob(source), "refund failure task");
    failedRefund.setGlobalJobRefund(refundGlobal);
    ok(failedRefund.acceptSourceJob(), "refund failure acceptance");
    failedRefund.proveJobFinished();
  }
  require(snapshot(source).phase == Phase::Poisoned && snapshot(source).operationFamilies == 1,
          "failed global disposition allowed source join");
  failRefund = false;
  root.reset();
}
} // namespace
int main() {
  try {
    exactRecipeValidation();
    rolesSurviveAliasesAndMemberDestruction();
    finalJobTailsAndKnownRejection();
    sealBeforeAcceptance();
    unknownWorkRetainsAccounting();
    std::cout << "PASS capture recipe CPU policy/custody: exact recipe, roles, aliases, "
                 "member/task tails, rejection, seal and poison\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL capture recipe: " << error.what() << '\n';
    return 1;
  }
}

#include "../include/minigpu_capture_recipe_owner.h"
#include <string_view>
#include <utility>

namespace mgpu::capture_recipe {
namespace {
Result layout(const SourceDomain &source, source_scope::Snapshot &out) noexcept {
  if (!source) return Result::Invalid;
  const auto result = source.snapshot(out);
  if (result != Result::Ok) return result;
  switch (out.phase) {
  case source_scope::Phase::Open: return Result::Ok;
  case source_scope::Phase::Sealed: return Result::Sealed;
  case source_scope::Phase::Joined: return Result::Joined;
  case source_scope::Phase::Poisoned: return Result::Poisoned;
  }
  return Result::Internal;
}
bool dimensionsMatch(const void *initial, uint32_t size,
                      const source_scope::SourceSpec &spec) noexcept {
  if (!initial || size != capture_recipe_v1::parameterBytes) return false;
  const auto *bytes = static_cast<const uint8_t *>(initial);
  const std::array<uint32_t, 4> words{spec.width, spec.height, 0, 0};
  for (size_t word = 0; word < words.size(); ++word)
    for (uint32_t byte = 0; byte < 4; ++byte)
      if (bytes[word * 4 + byte] != ((words[word] >> (byte * 8)) & 0xffu))
        return false;
  return true;
}
} // namespace

Result workStatus(const SourceDomain &source) noexcept {
  if (!source) return Result::Ok;
  source_scope::Snapshot snapshot;
  return layout(source, snapshot);
}
Result classifyResource(const SourceDomain &source, uint64_t bytes, uint32_t usage,
                        const void *initial, uint32_t initialBytes,
                        ResourceKind &role) noexcept {
  role = ResourceKind::Count;
  source_scope::Snapshot snapshot;
  const auto result = layout(source, snapshot);
  if (result != Result::Ok) return result;
  if (snapshot.spec.recipeRevision != capture_recipe_v1::revision) return Result::Invalid;
  if (!bytes && !usage && !initial && !initialBytes) role = ResourceKind::RecipeShader;
  else if (bytes == snapshot.frameBytes && usage == capture_recipe_v1::packedUsage &&
           !initial && !initialBytes) role = ResourceKind::PackedBuffer;
  else if (bytes == capture_recipe_v1::parameterBytes &&
           usage == capture_recipe_v1::parametersUsage &&
           dimensionsMatch(initial, initialBytes, snapshot.spec)) role = ResourceKind::Parameters;
  return role == ResourceKind::Count ? Result::Invalid : Result::Ok;
}
Result validateSetup(const SourceDomain &source, const SourceDomain &shaderSource,
                     ResourceKind shaderRole, const char *wgsl, size_t length,
                     uint32_t frameSlot, const std::array<Binding, 2> &bindings) noexcept {
  source_scope::Snapshot snapshot;
  const auto result = layout(source, snapshot);
  if (result != Result::Ok) return result;
  if (!sameSourceDomain(source, shaderSource) || shaderRole != ResourceKind::RecipeShader ||
      !wgsl || length != capture_recipe_v1::wgsl.size() ||
      std::string_view(wgsl, length) != capture_recipe_v1::wgsl ||
      frameSlot != capture_recipe_v1::frameSlot) return Result::Invalid;
  bool packed = false, parameters = false;
  for (const auto &binding : bindings) {
    if (!sameSourceDomain(source, binding.source) || binding.offset) return Result::Invalid;
    if (binding.slot == capture_recipe_v1::packedSlot && binding.kind == 1 &&
        binding.role == ResourceKind::PackedBuffer && binding.bytes == snapshot.frameBytes &&
        !packed) packed = true;
    else if (binding.slot == capture_recipe_v1::parametersSlot && binding.kind == 3 &&
             binding.role == ResourceKind::Parameters &&
             binding.bytes == capture_recipe_v1::parameterBytes && !parameters) parameters = true;
    else return Result::Invalid;
  }
  return packed && parameters ? Result::Ok : Result::Invalid;
}

ResourceCustody::ResourceCustody(ResourceCustody &&other) noexcept
    : source_(std::move(other.source_)), claim_(std::move(other.claim_)), metadata_(std::move(other.metadata_)),
      role_(std::exchange(other.role_, ResourceKind::Count)),
      accepted_(std::exchange(other.accepted_, false)),
      released_(other.released_.load()), uncertain_(other.uncertain_.load()) {}
ResourceCustody::~ResourceCustody() {
  if (!claim_) { metadata_.disposed(false); return; }
  Result result = Result::Invalid;
  if (!uncertain_.load(std::memory_order_acquire)) {
    if (!accepted_) result = claim_.cancelUnaccepted();
    else if (released_.load(std::memory_order_acquire)) result = claim_.retire();
  }
  if (result != Result::Ok) source_.quarantine();
  metadata_.disposed(result == Result::Ok);
  // A still-live claim's own destructor retains poisoned accounting.
}
ResourceCustody::ResourceCustody(ResourceCustody &&other,
                               source_scope::MetadataAllocation allocation) noexcept
    : ResourceCustody(std::move(other)) { metadata_ = std::move(allocation); }
Result ResourceCustody::reserveResourceMetadata(source_scope::MetadataReservations &out) noexcept {
  if (out.scoped()) return Result::Invalid;
  return source_ ? claim_.reserveResourceMetadata(out) : Result::Ok;
}
Result ResourceCustody::reserveSourceRole(const SourceDomain &source,
                                         ResourceKind role) noexcept {
  if (!source) return Result::Ok;
  if (source_ || claim_) return Result::Invalid;
  source_ = source;
  const auto result = source.reserve(role, claim_);
  if (result == Result::Ok) role_ = role;
  return result;
}
Result ResourceCustody::acceptSourceRole() noexcept {
  if (!source_) return Result::Ok;
  if (accepted_ || !claim_) return Result::Invalid;
  const auto result = claim_.accept();
  if (result == Result::Ok) accepted_ = true;
  return result;
}
void ResourceCustody::quarantineSourceRole() noexcept {
  uncertain_.store(true, std::memory_order_release);
  source_.quarantine();
}

JobCustody::~JobCustody() {
  bool known = !uncertain_.load(std::memory_order_acquire) &&
      (!accepted_ || finished_.load(std::memory_order_acquire));
  if (primary_ && known) {
    const auto result = accepted_ ? primary_.retire() : primary_.cancelUnaccepted();
    known = result == Result::Ok;
  }
  // Keep the operation family through global disposition too. A failed refund
  // cannot race a zero-obligation source join or turn into an invented receipt.
  if (globalRefund_ && known) known = globalRefund_();
  if (operation_ && known) known = operation_.finish() == Result::Ok;
  if (!known) source_.quarantine();
  metadata_.disposed(known);
}
JobCustody::JobCustody(JobCustody &&other, source_scope::MetadataAllocation allocation) noexcept
    : source_(std::move(other.source_)), operation_(std::move(other.operation_)),
      primary_(std::move(other.primary_)), metadata_(std::move(allocation)),
      accepted_(std::exchange(other.accepted_, false)),
      finished_(other.finished_.exchange(false)), uncertain_(other.uncertain_.exchange(false)),
      globalRefund_(std::exchange(other.globalRefund_, nullptr)) {}
Result JobCustody::reserveJobMetadata(source_scope::MetadataReservations &slots) const noexcept {
  return source_ ? operation_.view().reserveMetadata({source_scope::MetadataKind::PrimaryJob}, slots)
                 : Result::Ok;
}
Result JobCustody::prepareSourceJob(const SourceDomain &source) noexcept {
  if (!source) return Result::Ok;
  if (source_ || operation_ || primary_) return Result::Invalid;
  source_ = source;
  auto result = source.beginOperation(source_scope::OperationPurpose::Work, operation_);
  if (result != Result::Ok) return result;
  result = operation_.view().reserveContinuation(
      source_scope::ContinuationKind::PrimaryTask, primary_);
  if (result != Result::Ok) {
    if (operation_.finish() != Result::Ok) source_.quarantine();
  }
  return result;
}
Result JobCustody::acceptSourceJob() noexcept {
  if (accepted_ || (source_ && (!operation_ || !primary_))) return Result::Invalid;
  const auto status = workStatus(source_);
  if (status != Result::Ok) return status;
  const auto result = primary_ ? primary_.accept() : Result::Ok;
  if (result == Result::Ok) accepted_ = true;
  return result;
}
void JobCustody::quarantineSourceJob() noexcept {
  uncertain_.store(true, std::memory_order_release);
  source_.quarantine();
}
} // namespace mgpu::capture_recipe

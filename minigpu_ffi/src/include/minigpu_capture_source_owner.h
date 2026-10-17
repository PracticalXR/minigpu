#ifndef MINIGPU_CAPTURE_SOURCE_OWNER_H
#define MINIGPU_CAPTURE_SOURCE_OWNER_H
// The production CPU ownership seam; native create/release alone are adapters.
#include "minigpu_capture_source.h"
#include "minigpu_source_scope_owner.h"
#include <memory>

namespace mgpu::capture_source {
int validate(const MGPUCaptureSourceDesc *, uint32_t) noexcept;
uint32_t liveOwners() noexcept;
namespace detail { struct State; struct PreparedAnchor; }
using Owner = std::shared_ptr<detail::State>;
using Create = int (*)(void *, const MGPUCaptureSourceDesc &,
    source_scope::SourceRootOwner &, MGPUContextLease *&);
using Release = bool (*)(MGPUContextLease *) noexcept;
class PreparedSource {
public:
  PreparedSource();
  ~PreparedSource();
  PreparedSource(const PreparedSource &) = delete;
  int prepare(const MGPUCaptureSourceDesc &) noexcept;
  int prepareFunded(const MGPUCaptureSourceDesc &, setup_custody::SetupTicket state,
      setup_custody::SetupTicket anchor) noexcept;
  int create(void *original, Create, Release) noexcept;
  const Owner &owner() const noexcept { return owner_; }
  MGPUContextLease *borrowedLease() const noexcept;
  // Genuine local control-flow proof, not result-code/null-output inference by a host.
  bool knownUnpublished() const noexcept { return knownUnpublished_; }
private:
  int prepareImpl(const MGPUCaptureSourceDesc &, setup_custody::SetupTicket,
      setup_custody::SetupTicket, bool funded) noexcept;
  Owner owner_;
  setup_custody::SetupUnique<detail::PreparedAnchor> anchor_;
  bool attempted_ = false;
  bool knownUnpublished_ = false;
};
int getInfo(const Owner &, MGPUCaptureSourceInfo &) noexcept;
int seal(const Owner &) noexcept;
int tryReleaseJoined(const Owner &, MGPUCaptureSourceJoined &) noexcept;
// Private staged native route; not a public grant or original-budget bridge.
int createScopedFunded(MGPUContextLease *, const MGPUCaptureSourceDesc &,
    const setup_custody::SetupTickets &, MGPUCaptureSource **, MGPUContextLease **) noexcept;
} // namespace mgpu::capture_source
#endif

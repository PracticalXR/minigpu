#ifndef MINIGPU_CAPTURE_SETUP_BOXES_H
#define MINIGPU_CAPTURE_SETUP_BOXES_H
// Complete private CPU box types shared by real factories/layouts and cold
// empty-box witnesses. Never serialize/dereference these through guest code.
#include "minigpu_capture_source_owner.h"
namespace mgpu { class ContextPin; }
struct MGPUContextLease : mgpu::setup_custody::SetupUniqueRecord {
  explicit MGPUContextLease(mgpu::setup_custody::SetupUniqueRecord record = {}) noexcept
      : SetupUniqueRecord(std::move(record)) {}
  std::shared_ptr<mgpu::ContextPin> pin;
  bool sourceManaged = false;
};
struct MGPUCaptureSource : mgpu::setup_custody::SetupUniqueRecord {
  explicit MGPUCaptureSource(mgpu::setup_custody::SetupUniqueRecord record = {},
      mgpu::capture_source::Owner value = {}) noexcept
      : SetupUniqueRecord(std::move(record)), owner(std::move(value)) {}
  mgpu::capture_source::Owner owner;
};
namespace mgpu::setup_custody {
inline bool matchesOriginalRetain(const MGPUContextLease *original, const SetupTickets &tickets) noexcept {
  return original && original->setupTicket.hasRole(MGPU_CAPTURE_SETUP_ORIGINAL_LEASE) &&
      original->setupTicket.sameTransaction(tickets.roles[MGPU_CAPTURE_SETUP_ORIGINAL_LEASE]);
}
}
#endif

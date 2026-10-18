#ifndef MINIGPU_CREATION_IDENTITY_H
#define MINIGPU_CREATION_IDENTITY_H

// CPU-only equality policy shared by actual native owners and focused checks.
// Values are internal snapshots, never caller-issued capabilities or wire IDs.
#include "minigpu_source_scope_owner.h"

namespace mgpu {
struct RetainedCreationIdentity {
  const void *gpu = nullptr;
  const void *device = nullptr;
  const void *queue = nullptr;
  const void *instance = nullptr;
  uint64_t contextGeneration = 0;
  uint64_t deviceGeneration = 0;
  source_scope::SourceDomain source;
};

inline bool samePrivateCreation(const RetainedCreationIdentity &left,
                                const RetainedCreationIdentity &right) noexcept {
  if (!sameSourceDomain(left.source, right.source) || left.gpu != right.gpu ||
      left.device != right.device ||
      left.contextGeneration != right.contextGeneration ||
      left.deviceGeneration != right.deviceGeneration) return false;
  if (!left.source) return true; // Preserve legacy private-resource semantics.
  return left.gpu && left.device && left.queue && left.instance &&
      left.contextGeneration && left.deviceGeneration &&
      left.queue == right.queue && left.instance == right.instance;
}

inline bool sameRetainedCreation(const RetainedCreationIdentity &left,
                                 const RetainedCreationIdentity &right) noexcept {
  return samePrivateCreation(left, right) && left.queue == right.queue &&
      left.instance == right.instance;
}

inline bool sameQuarantinedCreation(const RetainedCreationIdentity &left,
                                    const RetainedCreationIdentity &right) noexcept {
  if (left.source || right.source) return sameRetainedCreation(left, right);
  // Existing unscoped quarantine coalesces creation generations. Distinct
  // scoped roots must never lose their pin claim through that legacy shortcut.
  return left.contextGeneration == right.contextGeneration &&
      left.deviceGeneration == right.deviceGeneration;
}
} // namespace mgpu

#endif

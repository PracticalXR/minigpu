#ifndef MINIGPU_SETUP_CUSTODY_ABI_H
#define MINIGPU_SETUP_CUSTODY_ABI_H
/* Private trusted host boundary, never guest/room/serialized data or consent.
 * Every native copy owns one raw ORIGINAL authority reference. Retain/release
 * affect only that lifetime; a terminal event never consumes its invoking copy.
 * The host independently pins callback code and the static ops table. Callbacks
 * must not throw/unwind or unload their own module, including final release.
 * Native10 allocation receipts are not source join or capture authority. */
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
enum MGPUSetupEventV1 {
  MGPU_SETUP_BEGIN_ALLOCATION = 0, MGPU_SETUP_ALLOCATED = 1,
  MGPU_SETUP_CONSTRUCTED = 2, MGPU_SETUP_DISPOSED = 3,
  MGPU_SETUP_BEGIN_DEALLOCATION = 4, MGPU_SETUP_FREE_RETURNED = 5,
  MGPU_SETUP_ALLOCATION_FAILED = 6, MGPU_SETUP_CANCEL_UNALLOCATED = 7,
  MGPU_SETUP_QUARANTINE = 8
};
typedef struct MGPUSetupEventRecordV1 {
  uint32_t struct_size, version, event, role;
  uint64_t generation;
  uint32_t row, alignment;
  uint64_t address, bytes;
} MGPUSetupEventRecordV1;
typedef struct MGPUSetupAuthorityOpsV1 {
  void (*retain)(void *);
  void (*release)(void *);
  uint32_t (*event)(void *, const MGPUSetupEventRecordV1 *);
} MGPUSetupAuthorityOpsV1;
typedef struct MGPUSetupExtentV1 {
  uint64_t bytes;
  uint32_t alignment, reserved;
} MGPUSetupExtentV1;
typedef struct MGPUSetupAuthorityV1 {
  uint32_t struct_size, version;
  void *context;
  const MGPUSetupAuthorityOpsV1 *ops;
  uint64_t generation;
  uint32_t row, role_count;
  uint64_t reserved;
  MGPUSetupExtentV1 roles[10];
} MGPUSetupAuthorityV1;
#ifdef __cplusplus
}
#include <cstddef>
static_assert(sizeof(MGPUSetupEventRecordV1) == 48 && offsetof(MGPUSetupEventRecordV1, address) == 32);
static_assert(sizeof(MGPUSetupExtentV1) == 16);
#if UINTPTR_MAX == UINT64_MAX
static_assert(sizeof(MGPUSetupAuthorityOpsV1) == 24);
static_assert(sizeof(MGPUSetupAuthorityV1) == 208 && offsetof(MGPUSetupAuthorityV1, roles) == 48);
#endif
#endif
#endif

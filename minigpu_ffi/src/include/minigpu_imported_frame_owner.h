#ifndef MINIGPU_IMPORTED_FRAME_OWNER_H
#define MINIGPU_IMPORTED_FRAME_OWNER_H
// Internal resource ownership, never guest/room transport or authority.
#include "minigpu_imported_frame.h"
#include "minigpu_context_owner.h"
#include "minigpu_capture_recipe_owner.h"
#include "minigpu_source_metadata_allocator.h"
#include <memory>
#include <mutex>
namespace mgpu {
struct ImportedFrameNative;
struct ImportedFrameState : capture_recipe::ResourceCustody {
  ImportedFrameState(source_scope::MetadataAllocation, capture_recipe::ResourceCustody &&,
                     source_scope::MetadataReservations &);
  ~ImportedFrameState();
  std::mutex mutex;
  std::shared_ptr<ContextPin> pin;
  uint32_t phase = MGPU_IMPORTED_FRAME_READY;
  int result = MGPU_CONTEXT_LEASE_OK;
  uint32_t width = 0, height = 0, registrySlot = 0;
  uint64_t bytes = 0;
  WGPUTextureView view = nullptr;
  source_scope::MetadataUnique<ImportedFrameNative> native;
};
std::shared_ptr<ImportedFrameState> copyImportedFrameState(MGPUImportedFrame*);
bool sameFrameCreation(const ContextPin&, const ContextPin&);
void quarantineImportedFrame(const std::shared_ptr<ImportedFrameState>&) noexcept;
// Called on creation worker after actual queue success, outside shader/frame/
// GPU locks. Phase must already be RETIRING. No retry after access attempt.
int finishImportedFrameRetirement(const std::shared_ptr<ImportedFrameState>&);
} // namespace mgpu
#endif

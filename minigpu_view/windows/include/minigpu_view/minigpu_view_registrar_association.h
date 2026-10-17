#ifndef MINIGPU_VIEW_REGISTRAR_ASSOCIATION_H_
#define MINIGPU_VIEW_REGISTRAR_ASSOCIATION_H_
#include "minigpu_view_private_preview.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct MGPUViewRegistrarAssociation MGPUViewRegistrarAssociation;
enum { MGPU_VIEW_ASSOCIATION_OPEN = 1, MGPU_VIEW_ASSOCIATION_CLOSED = 2 };
typedef struct MGPUViewRegistrarAssociationInfo {
  uint32_t struct_size, version, phase, platform_thread_id;
  uint64_t association_generation, reserved;
} MGPUViewRegistrarAssociationInfo; /*32 bytes, no wrapper pointer. */

/* Trusted native host only. The plugin privately publishes the actual pair at
 * registration. Acquire finds it; no public operation manufactures a pair.
 * core_texture_registrar is FlutterDesktopTextureRegistrarRef, NOT the DLL's
 * flutter::TextureRegistrar*. It must come from the retained original engine.
 * Acquire/Bind use its original platform thread. Other operations may use the
 * serialized source worker. Final release cannot race a call on that handle.
 * Bounds:16 states,128 acquisition handles; closed retained state stays charged.
 * State/handle/module retention is NOT Flutter wrapper/engine retention. */
MGPU_VIEW_PRIVATE_EXPORT int MGPUViewAcquireRegistrarAssociation(
    void *core_texture_registrar, MGPUViewRegistrarAssociation **);
MGPU_VIEW_PRIVATE_EXPORT int MGPUViewGetRegistrarAssociationInfo(
    MGPUViewRegistrarAssociation *, MGPUViewRegistrarAssociationInfo *, uint32_t out_size);
/* Caller already retains the actual engine root and keeps it through all host,
 * dispatcher, registration, callback and removal obligations. Independent
 * qualified engine/raster profile and original producer checks are unchanged.
 * The verifier's first argument is the original core C registrar, NOT the
 * DLL-local C++ wrapper. It must qualify that retained original engine/thread
 * profile; the plugin alone establishes the exact wrapper association.
 * At most one binding/host per association. No raw wrapper escapes the DLL.
 * The binding reserves before calling a foreign verifier; reentrant closure
 * wins and cannot publish a late host. No capture/private-ready grant is made.
 * Never serialize this interface, pointers or handles to Dart/guests/rooms. */
MGPU_VIEW_PRIVATE_EXPORT int MGPUViewBindAssociatedTrustedHost(
    MGPUViewRegistrarAssociation *, const MGPUViewHostBinding *, MGPUViewTrustedHost **);
MGPU_VIEW_PRIVATE_EXPORT int MGPUViewReleaseRegistrarAssociation(MGPUViewRegistrarAssociation *);
#ifdef __cplusplus
}
#endif
#endif

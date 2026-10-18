#ifndef MINIGPU_VIEW_PRIVATE_PREVIEW_H_
#define MINIGPU_VIEW_PRIVATE_PREVIEW_H_
#include <stdint.h>
#if defined(FLUTTER_PLUGIN_IMPL) || defined(MGPU_VIEW_PRIVATE_PREVIEW_IMPL)
#define MGPU_VIEW_PRIVATE_EXPORT __declspec(dllexport)
#else
#define MGPU_VIEW_PRIVATE_EXPORT __declspec(dllimport)
#endif
#ifdef __cplusplus
extern "C" {
#endif

typedef struct MGPUViewTrustedHost MGPUViewTrustedHost;
typedef struct MGPUViewPrivateRegistration MGPUViewPrivateRegistration;
typedef struct MGPUViewHostBinding MGPUViewHostBinding;
/* Trusted NATIVE host only. The verifier must authenticate the loaded engine
 * binary/profile, this actual registrar/engine instance and both live thread
 * identities against its independently qualified receipt. Neither a profile
 * enum, arbitrary pointer/ID nor a Dart/guest Boolean supplies that authority.
 * Return 0 only for the qualified distinct-thread Windows V1 profile: registrar
 * failure <0 creates no registration; descriptor calls are serialized on raster;
 * unregister callback there follows actual removal, while failed post calls it
 * inline on platform. Caller retains engine instance/registrar through all loans
 * and receipts; module pinning alone cannot keep that instance alive.
 * The first callback argument is the C++ wrapper for legacy BindTrustedHost,
 * or the original core C registrar for BindAssociatedTrustedHost. The latter
 * verifies the pair internally and never exports its DLL-local C++ wrapper. */
typedef int (*MGPUViewVerifyEngineProfile)(
    void *texture_registrar, const MGPUViewHostBinding *, void *engine_module);
struct MGPUViewHostBinding {
  uint32_t struct_size, version, profile, reserved;
  uint64_t engine_generation, qualification_receipt_id;
  void *engine_code_anchor, *original_minigpu_query;
  void *platform_thread_handle, *raster_thread_handle;
  MGPUViewVerifyEngineProfile verify;
  void *verifier_userdata;
}; /* V1 Win64: 80 bytes. profile=1; no method-channel serialization. */

enum MGPUViewPrivateResult {
  MGPU_VIEW_OK = 0, MGPU_VIEW_INVALID = -1, MGPU_VIEW_NOT_READY = -2,
  MGPU_VIEW_RETIRED = -3, MGPU_VIEW_UNSUPPORTED = -4,
  MGPU_VIEW_INTERNAL = -5, MGPU_VIEW_WOULD_BLOCK = -6,
  MGPU_VIEW_QUARANTINED = -7
};
enum MGPUViewPrivatePhase {
  MGPU_VIEW_CLOSED = 1, MGPU_VIEW_ACTIVE = 2, MGPU_VIEW_RETIRING = 3,
  MGPU_VIEW_REMOVED = 4, MGPU_VIEW_RELEASED = 5, MGPU_VIEW_QUARANTINE = 6
};
typedef struct MGPUViewPrivateInfo {
  uint32_t struct_size, version, phase;
  int32_t last_result;
  uint64_t registration_id, engine_generation, output_id, loan_id;
  int64_t texture_id;
  // registered stays 1 until confirmed removal; texture_id remains historical.
  uint32_t width, height, live_borrows, registered;
  uint64_t wake_event, wake_epoch;
} MGPUViewPrivateInfo; /* 88 bytes. Event is borrowed, never CloseHandle it. */
typedef struct MGPUViewPrivateReceipt {
  uint32_t struct_size, version;
  int32_t result;
  uint32_t reserved;
  uint64_t registration_id, engine_generation, output_id, loan_id;
} MGPUViewPrivateReceipt; /* 48 bytes. Ownership cleanup, not GPU/scanout proof. */
typedef struct MGPUViewPrivateStats {
  uint32_t struct_size, version, hosts, registrations, quarantined_registrations;
  uint32_t descriptor_borrows, max_hosts, max_registrations, max_descriptors;
  uint32_t reserved;
} MGPUViewPrivateStats; /* 40 bytes. Quarantine remains charged. */

/* All raw arguments are trusted host-local live pointers, from their original
 * modules. Final Release cannot race calls/waits on that same raw handle. No
 * source/capture/encode/input/private-ready permission is created. The SDK must
 * retain its original source, once-consumed consent and code through receipts.
 * texture_registrar is the host's actual flutter::TextureRegistrar*. The
 * verifier is trusted native code; module/address matching is not provenance.
 * Bind/Register/Begin/Close run on the qualified platform thread. No GPU init,
 * handle probe, fallback, library search, LRU eviction or timeout success.
 * Required modules, including callback code, are pinned for process lifetime.
 * First profile: 16 hosts, 256 registrations, 4 descriptors per registration. */
MGPU_VIEW_PRIVATE_EXPORT int MGPUViewBindTrustedHost(
    void *texture_registrar, const MGPUViewHostBinding *, MGPUViewTrustedHost **);
/* Caller retains its original MGPUPrivateOutput through synchronous return.
 * Register a closed empty surface BEFORE acquiring the native one-time loan.
 * A nonnull registration on error owns an accepted/ambiguous obligation and
 * must be retained/drained. Only a no-obligation refusal clears it. The texture
 * ID is UI data; raw output/host pointers never enter Dart/guest channels. */
MGPU_VIEW_PRIVATE_EXPORT int MGPUViewRegisterPrivateOutput(
    MGPUViewTrustedHost *, void *original_output, uint64_t expected_output_id,
    MGPUViewPrivateRegistration **, MGPUViewPrivateInfo *, uint32_t out_size);
MGPU_VIEW_PRIVATE_EXPORT int MGPUViewBeginPrivateRetirement(MGPUViewPrivateRegistration *);
/* Serialized owner-worker operation. WOULD_BLOCK means wait on the borrowed
 * auto-reset wake_event, then collect again. Keep owner alive through the wait.
 * No callback/elapsed time by itself permits release. Acknowledges actual
 * qualified removal + native loan release + explicit descriptor/variant cleanup;
 * safe callback bookkeeping may retain independent CPU state until return. */
MGPU_VIEW_PRIVATE_EXPORT int MGPUViewCollectPrivateRetirement(
    MGPUViewPrivateRegistration *, MGPUViewPrivateReceipt *, uint32_t out_size);
MGPU_VIEW_PRIVATE_EXPORT int MGPUViewGetPrivateRegistrationInfo(
    MGPUViewPrivateRegistration *, MGPUViewPrivateInfo *, uint32_t out_size);
MGPU_VIEW_PRIVATE_EXPORT int MGPUViewReleasePrivateRegistration(MGPUViewPrivateRegistration *);
/* Unexpected shutdown closes descriptor gates and roots unresolved owners in
 * quarantine; it does not pretend engine destruction proves image cleanup. */
MGPU_VIEW_PRIVATE_EXPORT int MGPUViewCloseTrustedHost(MGPUViewTrustedHost *);
MGPU_VIEW_PRIVATE_EXPORT int MGPUViewReleaseTrustedHost(MGPUViewTrustedHost *);
MGPU_VIEW_PRIVATE_EXPORT int MGPUViewGetPrivateStats(MGPUViewPrivateStats *, uint32_t out_size);
#ifdef __cplusplus
}
#endif
#endif

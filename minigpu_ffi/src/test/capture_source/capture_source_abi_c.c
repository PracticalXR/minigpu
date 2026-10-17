#include "../../include/minigpu_capture_source.h"
#include "../../include/minigpu_capture_setup.h"
#include <stddef.h>
typedef char capture_desc_size[(sizeof(MGPUCaptureSourceDesc) == 48) ? 1 : -1];
typedef char capture_desc_generation[(offsetof(MGPUCaptureSourceDesc, context_generation) == 24) ? 1 : -1];
typedef char capture_info_size[(sizeof(MGPUCaptureSourceInfo) == 64) ? 1 : -1];
typedef char capture_info_identity[(offsetof(MGPUCaptureSourceInfo, source_id) == 16) ? 1 : -1];
typedef char capture_joined_size[(sizeof(MGPUCaptureSourceJoined) == 32) ? 1 : -1];
typedef char capture_metadata_size[(sizeof(MGPUCaptureSourceLogicalMetadataProfileV1) == 32) ? 1 : -1];
typedef char capture_metadata_reserved[(offsetof(MGPUCaptureSourceLogicalMetadataProfileV1, reserved) == 24) ? 1 : -1];
typedef char capture_layout_extent_size[(sizeof(MGPUCaptureMetadataExtentV1) == 16) ? 1 : -1];
typedef char capture_layout_kind_size[(sizeof(MGPUCaptureMetadataKindLayoutV1) == 64) ? 1 : -1];
typedef char capture_layout_size[(sizeof(MGPUCaptureMetadataLayoutV1) == 2056) ? 1 : -1];
typedef char capture_layout_kinds[(offsetof(MGPUCaptureMetadataLayoutV1, kinds) == 72) ? 1 : -1];
typedef char capture_cpu_metadata_size[(sizeof(MGPUCaptureSourceCpuMetadataProfileV1) == 2112) ? 1 : -1];
typedef char capture_cpu_metadata_logical[(offsetof(MGPUCaptureSourceCpuMetadataProfileV1, logical) == 16) ? 1 : -1];
typedef char capture_cpu_metadata_layout[(offsetof(MGPUCaptureSourceCpuMetadataProfileV1, layout) == 48) ? 1 : -1];
typedef char capture_cpu_metadata_reserved[(offsetof(MGPUCaptureSourceCpuMetadataProfileV1, reserved) == 2104) ? 1 : -1];
typedef char capture_setup_extent_size[(sizeof(MGPUCaptureSetupExtentV1) == 24) ? 1 : -1];
typedef char capture_setup_size[(sizeof(MGPUCaptureSetupLayoutV1) == 312) ? 1 : -1];
typedef char capture_setup_kinds[(offsetof(MGPUCaptureSetupLayoutV1, kinds) == 72) ? 1 : -1];
typedef char funded_setup_size[(sizeof(MGPUCaptureFundedSetupProfileV1) == 2448) ? 1 : -1];
typedef char funded_setup_table[(offsetof(MGPUCaptureFundedSetupProfileV1, setup) == 16) ? 1 : -1];
typedef char funded_setup_metadata[(offsetof(MGPUCaptureFundedSetupProfileV1, metadata) == 328) ? 1 : -1];
typedef char funded_setup_reserved[(offsetof(MGPUCaptureFundedSetupProfileV1, reserved) == 2440) ? 1 : -1];
typedef char setup_create_outcome_size[(sizeof(MGPUCaptureSetupCreateOutcomeV1) == 64) ? 1 : -1];
typedef char setup_create_outcome_key[(offsetof(MGPUCaptureSetupCreateOutcomeV1, generation) == 16) ? 1 : -1];
typedef char setup_create_outcome_context[(offsetof(MGPUCaptureSetupCreateOutcomeV1, context_generation) == 32) ? 1 : -1];
int capture_source_c_abi(void) {
  return mgpuCaptureSourceCreateScoped && mgpuCaptureSourceGetInfo &&
    mgpuCaptureSourceSeal && mgpuCaptureSourceTryReleaseJoined &&
    mgpuCaptureSourceGetLogicalMetadataProfileV1 && mgpuCaptureSourceGetMetadataLayoutV1 &&
    mgpuCaptureSourceGetCpuMetadataProfileV1 && mgpuCaptureSourceGetSetupAllocationLayoutV1 &&
    mgpuCaptureSourceGetFundedSetupProfileV1 && mgpuDefaultContextRetainLeaseFundedV1 &&
    mgpuCaptureSourceCreateScopedFundedV1 && mgpuCaptureSourceCreateScopedFundedOutcomeV1;
}

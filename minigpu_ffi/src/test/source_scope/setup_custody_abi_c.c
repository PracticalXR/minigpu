#include "../../include/minigpu_setup_custody_abi.h"
#include <stddef.h>
_Static_assert(sizeof(MGPUSetupEventRecordV1) == 48, "event C ABI");
_Static_assert(offsetof(MGPUSetupEventRecordV1, address) == 32, "event address C ABI");
_Static_assert(sizeof(MGPUSetupExtentV1) == 16, "extent C ABI");
#if UINTPTR_MAX == UINT64_MAX
_Static_assert(sizeof(MGPUSetupAuthorityOpsV1) == 24, "ops C ABI");
_Static_assert(sizeof(MGPUSetupAuthorityV1) == 208, "authority C ABI");
_Static_assert(offsetof(MGPUSetupAuthorityV1, roles) == 48, "role array C ABI");
#endif

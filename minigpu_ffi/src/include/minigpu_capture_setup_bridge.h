#ifndef MINIGPU_CAPTURE_SETUP_BRIDGE_H
#define MINIGPU_CAPTURE_SETUP_BRIDGE_H
#include "minigpu_capture_setup.h"
#include "minigpu_setup_allocator.h"
namespace mgpu::setup_custody {
// Only stacks: validate the complete actual layout before retaining callbacks.
// The descriptor's live ORIGINAL host authority/ops is a trusted prerequisite,
// not authenticated by this function. No owner/budget/source is manufactured.
int retainAuthority(const MGPUSetupAuthorityV1 *, uint32_t, SetupTickets &) noexcept;
}
#endif

#include "../../include/minigpu_allocation_extent.h"

// An unsupported adapter must not even need the payload's size/constructor or
// instantiate any vendor-private shared control-block type.
struct Incomplete;
namespace layout = mgpu::allocation_layout;
constexpr auto unavailable = layout::sharedExtent<Incomplete, std::allocator<Incomplete>>();
constexpr auto defaultUnavailable = layout::defaultSharedExtent<Incomplete>();
static_assert(!layout::sharedLayoutSupported);
static_assert(unavailable.status == layout::Status::Unsupported &&
    unavailable.extent.bytes == 0 && unavailable.extent.alignment == 0);
static_assert(defaultUnavailable.status == layout::Status::Unsupported &&
    defaultUnavailable.extent.bytes == 0 && defaultUnavailable.extent.alignment == 0);
struct alignas(64) Aligned { unsigned char bytes[128]; };
constexpr auto unique = layout::uniqueExtent<Aligned>();
static_assert(unique.status == layout::Status::Known &&
    unique.extent.bytes == sizeof(Aligned) && unique.extent.alignment == alignof(Aligned));
int main() { return 0; }

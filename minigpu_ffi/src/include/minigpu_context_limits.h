#ifndef MINIGPU_CONTEXT_LIMITS_H
#define MINIGPU_CONTEXT_LIMITS_H

#include <cstdint>

namespace mgpu::native_limits {
inline constexpr uint32_t maxContexts = 64;
inline constexpr uint32_t maxContextLeases = 256;
} // namespace mgpu::native_limits

#endif

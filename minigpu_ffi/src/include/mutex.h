#pragma once

#include <mutex>


namespace mgpu {
// Native: alias to std::mutex so std::condition_variable works
using mutex = std::mutex;

// Lock helper aliases (same on both platforms)
template <typename... Mutexes>
using scoped_lock = std::scoped_lock<Mutexes...>;

template <typename Mtx>
using lock_guard = std::lock_guard<Mtx>;

template <typename Mtx>
using unique_lock = std::unique_lock<Mtx>;

} // namespace mgpu
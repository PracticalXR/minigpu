#pragma once

#include <chrono>
#include <thread>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#elif defined(_WIN32)
#include <windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <time.h>
#include <unistd.h>
#endif

namespace mgpu {

inline void platformSleep(int milliseconds, WGPUInstance instance) {
#ifdef __EMSCRIPTEN__
  // DO NOT USE emscripten_sleep() - it blocks the main thread!
  // Instead, yield control back to the browser
  if (milliseconds == 0) {
    // Just yield to browser event loop
    emscripten_sleep(0); // This is safe - just yields
  } else {
    // For non-zero sleeps, we should NOT block
    // Instead, the calling code should use async patterns
    // But if we must sleep, use a very short yield
    emscripten_sleep(0); // Just yield, don't actually sleep
  }
#elif defined(_WIN32)
  wgpuInstanceProcessEvents(instance);
  Sleep(milliseconds);
#elif defined(__unix__) || defined(__APPLE__)
  struct timespec ts;
  ts.tv_sec = milliseconds / 1000;
  ts.tv_nsec = (milliseconds % 1000) * 1000000;
  wgpuInstanceProcessEvents(instance);
  nanosleep(&ts, nullptr);
#else
  wgpuInstanceProcessEvents(instance);
  std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
#endif
}

// One poll step of a loop waiting on the GPU (a buffer map or a queue fence):
// the caller polls its condition, then calls this with how long it has waited.
// Young waits only yield (most readbacks land within a millisecond or two);
// older ones sleep ~0.25 ms per step so a long drain stops burning a core.
// Windows sleeps on a high-resolution waitable timer because Sleep(1) lasts a
// whole scheduler tick (~15.6 ms by default).
inline void platformWaitBackoff(long long waitedMicros) {
#ifdef __EMSCRIPTEN__
  (void)waitedMicros;
  emscripten_sleep(0);
#else
  if (waitedMicros < 2000) {
    std::this_thread::yield();
    return;
  }
#if defined(_WIN32)
  struct HighResolutionTimer {
    // 0x2 = CREATE_WAITABLE_TIMER_HIGH_RESOLUTION (Windows 10 1803+; older
    // systems refuse it and take the fallback below).
    HANDLE handle = CreateWaitableTimerExW(nullptr, nullptr, 0x00000002,
                                           TIMER_ALL_ACCESS);
    ~HighResolutionTimer() {
      if (handle) CloseHandle(handle);
    }
  };
  thread_local HighResolutionTimer timer;
  LARGE_INTEGER due;
  due.QuadPart = -2500; // relative, in 100 ns units: 0.25 ms
  if (timer.handle && SetWaitableTimer(timer.handle, &due, 0, nullptr, nullptr, FALSE) &&
      WaitForSingleObject(timer.handle, INFINITE) == WAIT_OBJECT_0) {
    return;
  }
  // No high-resolution timer: keep yielding, and only fall back to a full
  // tick once the wait is clearly long.
  Sleep(waitedMicros < 50000 ? 0 : 1);
#else
  std::this_thread::sleep_for(std::chrono::microseconds(250));
#endif
#endif
}

// Platform-specific WebGPU event processing
inline void processWebGPUEvents(WGPUInstance instance, WGPUDevice device) {
#ifdef __EMSCRIPTEN__
  emscripten_sleep(0);
#else
  // Native platforms (Dawn): Use device tick if available
  if (device) {
    wgpuDeviceTick(device);
  } else if (instance) {
    wgpuInstanceProcessEvents(instance);
  }
#endif
}

} // namespace mgpu
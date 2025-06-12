#pragma once

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
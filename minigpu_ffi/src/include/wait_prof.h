#pragma once
/* ===========================================================================
 * wait_prof.h — env-gated instrument for the BLOCKING WAITS and per-call
 * allocations on minigpu's transfer path.
 *
 * WHY IT EXISTS: transfer-path frame-time spikes are famously hard to attribute
 * because the cost does not stay where it is spent. `wgpuQueueWriteBuffer` is
 * asynchronous, so its device-side execution is charged to whichever call next
 * waits on the device; a spin-wait that gets descheduled is charged to whatever
 * stage happened to be running. The result is a spike that "moves between
 * stages" run to run and cannot be explained from the host-side stage timers a
 * consumer can see.
 *
 * What this adds is the one distinction those timers cannot make: for every
 * blocking wait it records the WALL TIME and the SPIN ITERATION COUNT. A wait
 * that is long because the GPU was genuinely busy spins many thousands of
 * times; a wait that is long because the OS descheduled the thread shows a long
 * wall time against a handful of iterations. That ratio is the discriminator.
 *
 * Everything here is behind one `waitProfEnabled()` bool that is cached on
 * first use, so with MGPU_WAIT_PROF unset the cost is a predictable branch.
 *
 *   MGPU_WAIT_PROF=1        enable collection + an atexit summary
 *   MGPU_WAIT_PROF_MS=<n>   also log EVERY individual event over n ms as it
 *                           happens (site, duration, spin iterations), so an
 *                           outlier can be lined up against the consumer's own
 *                           per-frame output instead of averaged away
 * ======================================================================== */

#include <chrono>
#include <cstdint>

namespace mgpu {

// Instrumented sites. Keep in sync with kWaitSiteNames in buffer.cpp.
enum WaitSite : int {
  WP_RD_LOCK = 0,   // readDirect: acquiring the device mutex
  WP_RD_FLUSH,      // readDirect: flushing the pending compute batch
  WP_RD_ALLOC,      // readDirect: staging-buffer destroy + create
  WP_RD_SUBMIT,     // readDirect: encoder + finish + submit
  WP_RD_MAP,        // readDirect: waiting for the map to complete
  WP_RD_COPY,       // readDirect: memcpy out of the mapped range
  WP_RD_EVENTS,     // readDirect: the trailing wgpuInstanceProcessEvents
  WP_RD_TOTAL,      // readDirect: whole call
  WP_WR_LOCK,       // writeBytesAt: acquiring the device mutex
  WP_WR_FLUSH,      // writeBytesAt: flushing the pending compute batch
  WP_WR_WRITE,      // writeBytesAt: wgpuQueueWriteBuffer itself
  WP_WR_TOTAL,      // writeBytesAt: whole call
  WP_RB_MAP,        // batched readback: waiting for the map to complete
  WP_Q_LAT,         // WebGPU worker thread: enqueue -> task start
  WP_DRAIN,         // drain_dawn_events_with_timeout: whole wait
  WP_SITE_COUNT
};

bool waitProfEnabled();
void waitProfNote(int site, long long us, long long iters);
void waitProfDump();

inline long long waitProfNowUs() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// RAII timer for a site with no meaningful iteration count.
struct WaitScope {
  int site;
  long long t0;
  long long iters;
  bool on;
  explicit WaitScope(int s) : site(s), t0(0), iters(0), on(waitProfEnabled()) {
    if (on) t0 = waitProfNowUs();
  }
  ~WaitScope() {
    if (on) waitProfNote(site, waitProfNowUs() - t0, iters);
  }
  WaitScope(const WaitScope &) = delete;
  WaitScope &operator=(const WaitScope &) = delete;
};

} // namespace mgpu

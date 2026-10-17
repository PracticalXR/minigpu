#ifndef MINIGPU_CAPTURE_CPU_QUEUE_H
#define MINIGPU_CAPTURE_CPU_QUEUE_H

#include "../include/minigpu_task_queue.h"
#include "../include/minigpu_context_lifetime.h"
#include <vector>

namespace mgpu::capture_cpu_test {
// Existing custody fixtures retain copyable invocation witnesses to inspect
// worker/closure tails. Allocate those witnesses BEFORE transfer. Queue order
// and allocation checks exercise the production Core directly, separately.
inline int enqueueWitness(std::vector<std::function<void()>> &tasks,
                          task_queue::Prepared &task) noexcept {
  try {
    task_queue::Core queue;
    auto execution = std::make_shared<task_queue::Execution>();
    std::function<void()> invoke = [execution] { execution->run(); };
    tasks.push_back(std::move(invoke));
    if (!queue.pushScoped(task)) { tasks.pop_back(); return MGPU_CONTEXT_LEASE_INVALID; }
    if (!queue.takeFront(*execution)) std::terminate();
    return MGPU_CONTEXT_LEASE_OK;
  } catch (...) { return MGPU_CONTEXT_LEASE_INTERNAL; } // Before token transfer.
}
}
#endif

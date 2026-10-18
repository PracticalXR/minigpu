#ifndef MGPU_VIEW_PRIVATE_PREVIEW_IMPL
#define MGPU_VIEW_PRIVATE_PREVIEW_IMPL
#endif
#include "include/minigpu_view/minigpu_view_private_dispatch.h"
#include "private_preview_dispatcher.h"
#include "private_owner_close.h"
#include <array>
#include <limits>
#include <mutex>

namespace minigpu_view::private_dispatch {
using namespace private_preview;
constexpr size_t maxDispatchers = 16, maxSources = 32, maxRequests = 128;
constexpr UINT wakeMessage = WM_APP + 0x3a7;
constexpr wchar_t windowClass[] = L"MinigpuView.PrivatePreviewDispatch.v1";
struct Dispatcher;
struct Source;
struct Job;
struct Dispatcher {
  std::mutex gate;
  std::shared_ptr<Host> host;
  std::array<std::shared_ptr<Job>, maxRequests> queue{};
  size_t queued = 0, slot = 0;
  uint64_t lastEpoch = 0;
  HWND window = nullptr;
  bool closed = false;
  std::atomic<uint32_t> sources{0}, requests{0}, executing{0};
};
struct Source {
  std::shared_ptr<Dispatcher> dispatcher;
  std::shared_ptr<SourceFence> fence;
  size_t slot = 0;
  std::atomic<uint32_t> requests{0};
};
struct Job {
  std::mutex gate;
  std::shared_ptr<Source> source;
  MGPUViewPrivateDispatchInfo info{};
  MGPUViewPrivateInfo preview{};
  MGPUViewPrivateRegistration *registration = nullptr, *retiring = nullptr;
  void *output = nullptr, *userdata = nullptr;
  uint64_t expectedOutput = 0;
  MGPUViewPrivateDispatchCallback callback = nullptr;
  HANDLE wake = nullptr;
  size_t slot = 0;
  ~Job() { if (wake) CloseHandle(wake); }
};
struct Registry {
  std::mutex gate;
  std::array<std::shared_ptr<Dispatcher>, maxDispatchers> dispatchers{};
  std::array<std::shared_ptr<Source>, maxSources> sources{};
  std::array<std::shared_ptr<Job>, maxRequests> requests{};
  uint64_t nextRequest = 1;
};
Registry &registry() { static auto *r = new Registry; return *r; }

void destroyClosedWindow(const std::shared_ptr<Dispatcher> &d) noexcept {
  HWND window = nullptr;
  {
    std::lock_guard<std::mutex> lock(d->gate);
    if (d->closed && !d->executing.load()) window = d->window;
  }
  if (window && PrivateHostPlatformCurrent(d->host, false)) DestroyWindow(window);
  // WM_NCDESTROY alone clears the handle. Failed destruction retains the owner.
}

// An accepted job remains rooted through actual callback return, including
// shutdown cancellation. No foreign call executes with job/queue/registry held.
void execute(const std::shared_ptr<Job> &job, bool cancel) noexcept {
  auto d = job->source->dispatcher;
  int result = MGPU_VIEW_INTERNAL;
  bool callbackReturned = false;
  try {
    { std::lock_guard<std::mutex> lock(job->gate); job->info.phase = MGPU_VIEW_DISPATCH_EXECUTING; }
    bool closed;
    { std::lock_guard<std::mutex> lock(d->gate); closed = d->closed; }
    if (cancel || closed || !PrivateHostOpen(d->host)) result = MGPU_VIEW_RETIRED;
    else if (job->info.kind == MGPU_VIEW_DISPATCH_REGISTER) {
      result = RegisterPrivateSourceOutput(d->host, job->source->fence,
          job->output, job->expectedOutput, &job->registration, &job->preview,
          sizeof(job->preview));
    } else {
      result = MGPUViewBeginPrivateRetirement(job->retiring);
      MGPUViewGetPrivateRegistrationInfo(job->retiring, &job->preview, sizeof(job->preview));
    }
    {
      std::lock_guard<std::mutex> lock(job->gate);
      job->info.result = result;
      job->info.registration_id = job->preview.registration_id;
      job->info.phase = MGPU_VIEW_DISPATCH_CALLBACK;
    }
    job->callback(job->source->fence->epoch, job->info.request_id, job->userdata);
    callbackReturned = true;
  } catch (...) { /* No automatic release on an uncertain native/callback path. */ }
  {
    std::lock_guard<std::mutex> lock(job->gate);
    job->info.phase = callbackReturned ? MGPU_VIEW_DISPATCH_COMPLETE : MGPU_VIEW_DISPATCH_QUARANTINE;
    if (!callbackReturned) job->info.result = MGPU_VIEW_QUARANTINED;
    SetEvent(job->wake);
  }
  --d->executing;
  destroyClosedWindow(d);
}

LRESULT CALLBACK windowProcedure(HWND window, UINT message, WPARAM w, LPARAM l) {
  auto *raw = reinterpret_cast<Dispatcher *>(GetWindowLongPtrW(window, GWLP_USERDATA));
  if (message == WM_NCCREATE) {
    raw = static_cast<Dispatcher *>(reinterpret_cast<CREATESTRUCTW *>(l)->lpCreateParams);
    SetWindowLongPtrW(window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(raw));
  }
  if (message == WM_NCDESTROY && raw) {
    std::lock_guard<std::mutex> lock(raw->gate);
    raw->window = nullptr; SetWindowLongPtrW(window, GWLP_USERDATA, 0);
  }
  if (message != wakeMessage || !raw) return DefWindowProcW(window, message, w, l);
  try {
    std::shared_ptr<Job> job;
    {
      std::lock_guard<std::mutex> lock(raw->gate);
      if (raw->closed || !raw->queued) return 0;
      job = std::move(raw->queue[0]);
      for (size_t i = 1; i < raw->queued; ++i) raw->queue[i - 1] = std::move(raw->queue[i]);
      raw->queue[--raw->queued].reset(); ++raw->executing;
    }
    execute(job, false);
  } catch (...) { /* Existing bounded roots retain any uncertain queued work. */ }
  return 0;
}

bool registerWindowClass(HINSTANCE &module) {
  if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
      reinterpret_cast<LPCWSTR>(&windowProcedure), &module)) return false;
  WNDCLASSEXW type{}; type.cbSize = sizeof(type); type.hInstance = module;
  type.lpfnWndProc = windowProcedure; type.lpszClassName = windowClass;
  if (RegisterClassExW(&type)) return true;
  if (GetLastError() != ERROR_CLASS_ALREADY_EXISTS) return false;
  WNDCLASSEXW existing{}; existing.cbSize = sizeof(existing);
  return GetClassInfoExW(module, windowClass, &existing) &&
      existing.hInstance == module && existing.lpfnWndProc == windowProcedure;
}

void close(const std::shared_ptr<Dispatcher> &d) {
  std::array<std::shared_ptr<Job>, maxRequests> pending{};
  size_t count = 0;
  {
    std::lock_guard<std::mutex> lock(d->gate);
    d->closed = true;
    count = d->queued; pending.swap(d->queue); d->queued = 0;
    d->executing += static_cast<uint32_t>(count);
  }
  std::array<std::shared_ptr<Source>, maxSources> sources{};
  { auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate); sources = r.sources; }
  for (const auto &s : sources) if (s && s->dispatcher == d) {
    s->fence->closed = true;
    QuarantinePrivateSource(s->fence);
  }
  for (size_t i = 0; i < count; ++i) execute(pending[i], true);
  destroyClosedWindow(d);
}

void unrootJob(const std::shared_ptr<Job> &job) {
  auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate);
  r.requests[job->slot].reset();
  --job->source->requests; --job->source->dispatcher->requests;
}
} // namespace minigpu_view::private_dispatch

struct MGPUViewPrivateDispatcher { std::shared_ptr<minigpu_view::private_dispatch::Dispatcher> state; };
struct MGPUViewPrivateSource { std::shared_ptr<minigpu_view::private_dispatch::Source> state; };
struct MGPUViewPrivateDispatch { std::shared_ptr<minigpu_view::private_dispatch::Job> state; };
using namespace minigpu_view::private_preview;
using namespace minigpu_view::private_dispatch;

namespace minigpu_view::private_dispatch {
std::shared_ptr<Dispatcher> RetainPrivateDispatcher(MGPUViewPrivateDispatcher *h) {
  return h ? h->state : nullptr;
}
int ReservePrivateOwnerPair(const std::shared_ptr<Host> &h, const std::shared_ptr<Dispatcher> &d,
                           MGPUViewRegistrarAssociation *association, uint64_t request) {
  if (!d) return ReservePrivateHostClose(h, association, request, 0);
  std::lock_guard<std::mutex> lock(d->gate);
  if (!h || d->host != h) return MGPU_VIEW_INVALID;
  if (d->sources || d->requests || d->executing || d->queued) return MGPU_VIEW_WOULD_BLOCK;
  const auto result = ReservePrivateHostClose(h, association, request, 1);
  if (result == MGPU_VIEW_OK) d->closed = true; // Reject source/post admission before waking platform.
  return result;
}
int CloseRetainedPrivateDispatcher(const std::shared_ptr<Dispatcher> &d) {
  if (!d) return MGPU_VIEW_OK;
  if (!PrivateHostPlatformCurrent(d->host, false)) return MGPU_VIEW_WOULD_BLOCK;
  close(d);
  std::lock_guard<std::mutex> lock(d->gate);
  return !d->window && !d->executing ? MGPU_VIEW_OK : MGPU_VIEW_WOULD_BLOCK;
}
}

extern "C" int MGPUViewCreatePrivateDispatcher(MGPUViewTrustedHost *host,
    MGPUViewPrivateDispatcher **out) {
  if (out) *out = nullptr;
  if (!host || !out) return MGPU_VIEW_INVALID;
  try {
    auto retained = RetainPrivateHost(host);
    if (!PrivateHostOpen(retained)) return MGPU_VIEW_RETIRED;
    if (!PrivateHostPlatformCurrent(retained)) return MGPU_VIEW_WOULD_BLOCK;
    HINSTANCE module = nullptr;
    if (!registerWindowClass(module)) return MGPU_VIEW_INTERNAL;
    auto d = std::make_shared<Dispatcher>(); d->host = retained;
    auto box = std::make_unique<MGPUViewPrivateDispatcher>(); box->state = d;
    bool reserved = false;
    {
      auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate);
      for (const auto &other : r.dispatchers)
        if (other && other->host == retained) return MGPU_VIEW_WOULD_BLOCK;
      for (size_t i = 0; i < maxDispatchers; ++i) if (!r.dispatchers[i]) {
        if (!TryPrivateHostAddChild(retained)) return MGPU_VIEW_RETIRED;
        d->slot = i; r.dispatchers[i] = d;
        reserved = true; break;
      }
    }
    if (!reserved) return MGPU_VIEW_WOULD_BLOCK;
    d->window = CreateWindowExW(0, windowClass, L"", 0, 0, 0, 0, 0,
                               HWND_MESSAGE, nullptr, module, d.get());
    if (!d->window) {
      auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate);
      r.dispatchers[d->slot].reset(); PrivateHostRemoveChild(retained);
      return MGPU_VIEW_INTERNAL;
    }
    *out = box.release(); return MGPU_VIEW_OK;
  } catch (...) { return MGPU_VIEW_INTERNAL; }
}

extern "C" int MGPUViewOpenPrivateSource(MGPUViewPrivateDispatcher *handle,
    const MGPUViewPrivateSourceBinding *b, MGPUViewPrivateSource **out) {
  if (out) *out = nullptr;
  if (!handle || !b || !out || b->struct_size != sizeof(*b) || b->version != 1 ||
      !b->source_epoch || !b->valid_until_tick_ms || b->reserved || !b->original_minigpu_query)
    return MGPU_VIEW_INVALID;
  try {
    auto d = handle->state;
    if (b->original_minigpu_query != PrivateHostOriginalQuery(d->host)) return MGPU_VIEW_UNSUPPORTED;
    if (GetTickCount64() >= b->valid_until_tick_ms) return MGPU_VIEW_RETIRED;
    auto s = std::make_shared<Source>(); s->dispatcher = d;
    s->fence = std::make_shared<SourceFence>(b->source_epoch, b->valid_until_tick_ms);
    auto box = std::make_unique<MGPUViewPrivateSource>(); box->state = s;
    std::lock_guard<std::mutex> lock(d->gate);
    if (d->closed || !PrivateHostOpen(d->host)) return MGPU_VIEW_RETIRED;
    if (b->source_epoch <= d->lastEpoch) return MGPU_VIEW_RETIRED;
    auto &r = registry(); std::lock_guard<std::mutex> global(r.gate);
    for (size_t i = 0; i < maxSources; ++i) if (!r.sources[i]) {
      s->slot = i; r.sources[i] = s; ++d->sources; d->lastEpoch = b->source_epoch;
      *out = box.release(); return MGPU_VIEW_OK;
    }
    return MGPU_VIEW_WOULD_BLOCK;
  } catch (...) { return MGPU_VIEW_INTERNAL; }
}
extern "C" int MGPUViewFencePrivateSource(MGPUViewPrivateSource *handle) {
  if (!handle) return MGPU_VIEW_INVALID;
  handle->state->fence->closed = true;
  return MGPU_VIEW_OK;
}

namespace {
int post(MGPUViewPrivateSource *handle, void *output, uint64_t expected,
    MGPUViewPrivateRegistration *retiring, MGPUViewPrivateDispatchCallback callback,
    void *userdata, MGPUViewPrivateDispatch **out) {
  if (out) *out = nullptr;
  if (!handle || !callback || !out || (!retiring && (!output || !expected))) return MGPU_VIEW_INVALID;
  try {
    auto s = handle->state; auto d = s->dispatcher;
    if (retiring) {
      if (!PrivateRegistrationMatchesSource(retiring, s->fence)) return MGPU_VIEW_INVALID;
    } else if (!s->fence->current()) return MGPU_VIEW_RETIRED;
    HMODULE callbackModule = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
        reinterpret_cast<LPCWSTR>(callback), &callbackModule)) return MGPU_VIEW_UNSUPPORTED;
    auto job = std::make_shared<Job>(); job->source = s; job->output = output;
    job->expectedOutput = expected; job->retiring = retiring;
    job->callback = callback; job->userdata = userdata;
    job->wake = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!job->wake) return MGPU_VIEW_INTERNAL;
    auto box = std::make_unique<MGPUViewPrivateDispatch>(); box->state = job;
    // Acquire every fallible lock before rooting a request. This is the same
    // dispatcher -> registry order as OpenSource. No lock acquisition, foreign
    // callback or allocation may intervene after native ownership is accepted.
    std::lock_guard<std::mutex> queue(d->gate);
    if (d->closed || !d->window || !PrivateHostOpen(d->host) ||
        (!retiring && !s->fence->current())) return MGPU_VIEW_RETIRED;
    auto &r = registry(); std::lock_guard<std::mutex> global(r.gate);
    if (s->requests.load() >= 4 || d->queued >= d->queue.size() ||
        r.nextRequest == (std::numeric_limits<uint64_t>::max)()) return MGPU_VIEW_WOULD_BLOCK;
    size_t slot = maxRequests;
    for (size_t i = 0; i < maxRequests; ++i) if (!r.requests[i]) { slot = i; break; }
    if (slot == maxRequests) return MGPU_VIEW_WOULD_BLOCK;
    job->slot = slot;
    job->info = {sizeof(job->info), 1,
        static_cast<uint32_t>(retiring ? MGPU_VIEW_DISPATCH_RETIRE : MGPU_VIEW_DISPATCH_REGISTER),
        MGPU_VIEW_DISPATCH_QUEUED, MGPU_VIEW_NOT_READY, 0,
        r.nextRequest++, s->fence->epoch, 0, reinterpret_cast<uintptr_t>(job->wake)};
    r.requests[slot] = job; ++s->requests; ++d->requests;
    d->queue[d->queued++] = job;
    // Hold both locks through post/rollback and output publication. PostMessageW
    // does not invoke the target procedure inline. A synchronous refusal leaves
    // neither queue/registry ownership nor a callback; rollback takes no new lock.
    if (!PostMessageW(d->window, wakeMessage, 0, 0)) {
      d->queue[--d->queued].reset(); r.requests[slot].reset();
      --s->requests; --d->requests; return MGPU_VIEW_INTERNAL;
    }
    *out = box.release(); return MGPU_VIEW_OK;
  } catch (...) { return MGPU_VIEW_INTERNAL; }
}
}
extern "C" int MGPUViewPostPrivateRegister(MGPUViewPrivateSource *source,
    void *output, uint64_t expected, MGPUViewPrivateDispatchCallback callback,
    void *userdata, MGPUViewPrivateDispatch **out) {
  return post(source, output, expected, nullptr, callback, userdata, out);
}
extern "C" int MGPUViewPostPrivateRetirement(MGPUViewPrivateSource *source,
    MGPUViewPrivateRegistration *registration, MGPUViewPrivateDispatchCallback callback,
    void *userdata, MGPUViewPrivateDispatch **out) {
  if (out) *out = nullptr;
  if (!registration) return MGPU_VIEW_INVALID;
  return post(source, nullptr, 0, registration, callback, userdata, out);
}
extern "C" int MGPUViewGetPrivateDispatchInfo(MGPUViewPrivateDispatch *handle,
    MGPUViewPrivateDispatchInfo *out, uint32_t size) {
  if (!out || size < sizeof(*out)) return MGPU_VIEW_INVALID;
  *out = {};
  if (!handle) return MGPU_VIEW_INVALID;
  try {
    auto job = handle->state; std::lock_guard<std::mutex> lock(job->gate);
    *out = job->info; return MGPU_VIEW_OK;
  } catch (...) { return MGPU_VIEW_INTERNAL; }
}
extern "C" int MGPUViewCollectPrivateDispatch(MGPUViewPrivateDispatch *handle,
    MGPUViewPrivateDispatchInfo *info, uint32_t infoSize, MGPUViewPrivateRegistration **out,
    MGPUViewPrivateInfo *preview, uint32_t previewSize) {
  if (out) *out = nullptr;
  if (!info || infoSize < sizeof(*info) || !preview || previewSize < sizeof(*preview))
    return MGPU_VIEW_INVALID;
  *info = {}; *preview = {};
  if (!handle || !out) return MGPU_VIEW_INVALID;
  try {
    auto job = handle->state; std::lock_guard<std::mutex> lock(job->gate);
    if (job->info.phase == MGPU_VIEW_DISPATCH_QUARANTINE) return MGPU_VIEW_QUARANTINED;
    if (job->info.phase == MGPU_VIEW_DISPATCH_COLLECTED) return MGPU_VIEW_RETIRED;
    if (job->info.phase != MGPU_VIEW_DISPATCH_COMPLETE) return MGPU_VIEW_WOULD_BLOCK;
    job->info.phase = MGPU_VIEW_DISPATCH_COLLECTED;
    *info = job->info; *preview = job->preview;
    *out = job->registration; job->registration = nullptr;
    return MGPU_VIEW_OK;
  } catch (...) { return MGPU_VIEW_INTERNAL; }
}
extern "C" int MGPUViewReleasePrivateDispatch(MGPUViewPrivateDispatch *handle) {
  if (!handle) return MGPU_VIEW_INVALID;
  try {
    auto job = handle->state;
    { std::lock_guard<std::mutex> lock(job->gate);
      if (job->info.phase != MGPU_VIEW_DISPATCH_COLLECTED) return MGPU_VIEW_WOULD_BLOCK;
    }
    unrootJob(job); delete handle; return MGPU_VIEW_OK;
  } catch (...) { return MGPU_VIEW_INTERNAL; }
}
extern "C" int MGPUViewReleasePrivateSource(MGPUViewPrivateSource *handle) {
  if (!handle) return MGPU_VIEW_INVALID;
  try {
    auto s = handle->state;
    if (s->fence->current() || s->requests.load() || s->fence->registrations.load())
      return MGPU_VIEW_WOULD_BLOCK;
    auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate);
    r.sources[s->slot].reset(); --s->dispatcher->sources;
    delete handle; return MGPU_VIEW_OK;
  } catch (...) { return MGPU_VIEW_INTERNAL; }
}
extern "C" int MGPUViewClosePrivateDispatcher(MGPUViewPrivateDispatcher *handle) {
  if (!handle) return MGPU_VIEW_INVALID;
  try {
    auto d = handle->state;
    if (!PrivateHostPlatformCurrent(d->host, false)) return MGPU_VIEW_WOULD_BLOCK;
    close(d); return MGPU_VIEW_OK;
  } catch (...) { return MGPU_VIEW_INTERNAL; }
}
extern "C" int MGPUViewReleasePrivateDispatcher(MGPUViewPrivateDispatcher *handle) {
  if (!handle) return MGPU_VIEW_INVALID;
  try {
    auto d = handle->state;
    { std::lock_guard<std::mutex> lock(d->gate);
      if (!d->closed || d->window || d->sources.load() || d->requests.load() || d->executing.load())
        return MGPU_VIEW_WOULD_BLOCK;
    }
    auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate);
    r.dispatchers[d->slot].reset(); PrivateHostRemoveChild(d->host);
    delete handle; return MGPU_VIEW_OK;
  } catch (...) { return MGPU_VIEW_INTERNAL; }
}
namespace minigpu_view {
void ClosePrivatePreviewDispatchers(flutter::TextureRegistrar *registrar) noexcept {
  try {
    std::array<std::shared_ptr<private_dispatch::Dispatcher>, private_dispatch::maxDispatchers> states{};
    { auto &r = private_dispatch::registry(); std::lock_guard<std::mutex> lock(r.gate); states = r.dispatchers; }
    for (const auto &d : states) if (d && private_preview::PrivateHostRegistrar(d->host, registrar)) {
      if (private_preview::PrivateHostPlatformCurrent(d->host, false)) private_dispatch::close(d);
      // Off-profile destruction cannot use the registrar/window. Close every
      // descriptor gate and retain roots; the owner shutdown handles quarantine.
      else {
        std::lock_guard<std::mutex> lock(d->gate); d->closed = true;
      }
    }
  } catch (...) { /* Bounded existing roots retain unresolved requests/owners. */ }
}
}

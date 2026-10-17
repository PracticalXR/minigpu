#ifndef MGPU_VIEW_PRIVATE_PREVIEW_IMPL
#define MGPU_VIEW_PRIVATE_PREVIEW_IMPL
#endif
#include "include/minigpu_view/minigpu_view_private_owner_close.h"
#include "private_owner_close.h"
#include <array>
#include <limits>
#include <mutex>

namespace minigpu_view::private_close {
using namespace private_preview;
using namespace private_dispatch;
constexpr size_t maxOwners = 32;
constexpr UINT wakeMessage = WM_APP + 0x3a9;
constexpr wchar_t windowClass[] = L"MinigpuView.PrivateOwnerClose.v1";
struct State {
  std::mutex gate;
  MGPUViewRegistrarAssociation *association = nullptr;
  std::shared_ptr<Host> host;
  std::shared_ptr<Dispatcher> dispatcher;
  uint64_t id = 0, generation = 0;
  DWORD platform = 0;
  size_t slot = 0;
  HWND window = nullptr;
  HANDLE wake = nullptr;
  uint32_t phase = MGPU_VIEW_OWNER_CLOSE_READY, executing = 0;
  int result = MGPU_VIEW_OK;
  ~State() {
    // Only CPU metadata/kernel handles. Host/window cleanup is explicit above
    // this lifetime; failed owners stay rooted and never reach this destructor.
    if (association) MGPUViewReleaseRegistrarAssociation(association);
    if (wake) CloseHandle(wake);
  }
};
struct Registry {
  std::mutex gate;
  std::array<std::shared_ptr<State>, maxOwners> states{};
  uint64_t next = 1;
};
Registry &registry() { static auto *r = new Registry; return *r; }
MGPUViewPrivateOwnerCloseInfo info(const State &s) {
  return {sizeof(MGPUViewPrivateOwnerCloseInfo), 1, s.phase, s.result,
      s.id, s.generation, reinterpret_cast<uintptr_t>(s.wake),
      s.host ? 1u : 0u, s.dispatcher ? 1u : 0u};
}
void execute(const std::shared_ptr<State> &s) noexcept {
  int result = MGPU_VIEW_INTERNAL;
  try {
    {
      std::lock_guard<std::mutex> lock(s->gate);
      if (s->phase != MGPU_VIEW_OWNER_CLOSE_QUEUED) return;
      s->phase = MGPU_VIEW_OWNER_CLOSE_EXECUTING; ++s->executing;
    }
    if (GetCurrentThreadId() == s->platform) {
      result = CloseRetainedPrivateDispatcher(s->dispatcher);
      if (result == MGPU_VIEW_OK) result = CloseRetainedPrivateHost(s->host);
      if (result == MGPU_VIEW_OK) {
        HWND window = nullptr;
        { std::lock_guard<std::mutex> lock(s->gate); window = s->window; }
        if (!window || !DestroyWindow(window)) result = MGPU_VIEW_INTERNAL;
      }
    }
  } catch (...) { /* No successful receipt on unknown execution. */ }
  try {
    std::lock_guard<std::mutex> lock(s->gate);
    if (s->executing) --s->executing;
    const bool done = result == MGPU_VIEW_OK && !s->window && !s->executing;
    s->phase = done ? MGPU_VIEW_OWNER_CLOSE_COMPLETE : MGPU_VIEW_OWNER_CLOSE_FAILED;
    s->result = done ? MGPU_VIEW_OK : MGPU_VIEW_QUARANTINED;
    SetEvent(s->wake);
  } catch (...) { /* Existing bounded root retains the full uncertain operation. */ }
}
LRESULT CALLBACK windowProcedure(HWND window, UINT message, WPARAM w, LPARAM l) {
  auto *raw = reinterpret_cast<State *>(GetWindowLongPtrW(window, GWLP_USERDATA));
  if (message == WM_NCCREATE) {
    raw = static_cast<State *>(reinterpret_cast<CREATESTRUCTW *>(l)->lpCreateParams);
    SetWindowLongPtrW(window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(raw));
    std::lock_guard<std::mutex> lock(raw->gate); raw->window = window;
  }
  if (message == WM_NCDESTROY && raw) {
    std::lock_guard<std::mutex> lock(raw->gate);
    raw->window = nullptr; SetWindowLongPtrW(window, GWLP_USERDATA, 0);
  }
  if (message != wakeMessage || !raw) return DefWindowProcW(window, message, w, l);
  try {
    std::shared_ptr<State> retained;
    { auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate);
      if (raw->slot < maxOwners && r.states[raw->slot].get() == raw) retained = r.states[raw->slot]; }
    if (retained) execute(retained);
  } catch (...) { /* No pointer/command is accepted from the message payload. */ }
  return 0;
}
bool registerClass(HINSTANCE &module) {
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
}
struct MGPUViewPrivateOwnerClose { std::shared_ptr<minigpu_view::private_close::State> state; };
using namespace minigpu_view::private_close;

extern "C" int MGPUViewCreatePrivateOwnerClose(MGPUViewRegistrarAssociation *association,
    MGPUViewPrivateOwnerClose **out, MGPUViewPrivateOwnerCloseInfo *details, uint32_t size) {
  if (out) *out = nullptr;
  if (!details || size < sizeof(*details)) return MGPU_VIEW_INVALID;
  *details = {};
  if (!association || !out) return MGPU_VIEW_INVALID;
  std::shared_ptr<State> s;
  std::unique_ptr<MGPUViewPrivateOwnerClose> box;
  bool rooted = false;
  try {
    s = std::make_shared<State>();
    box = std::make_unique<MGPUViewPrivateOwnerClose>(); box->state = s;
    const auto retained = minigpu_view::RetainRegistrarAssociationForClose(association, &s->association);
    if (retained != MGPU_VIEW_OK) return retained;
    MGPUViewRegistrarAssociationInfo binding{};
    if (MGPUViewGetRegistrarAssociationInfo(s->association, &binding, sizeof(binding)) != MGPU_VIEW_OK ||
        binding.phase != MGPU_VIEW_ASSOCIATION_OPEN || binding.platform_thread_id != GetCurrentThreadId())
      return MGPU_VIEW_RETIRED;
    s->generation = binding.association_generation; s->platform = binding.platform_thread_id;
    s->wake = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!s->wake) return MGPU_VIEW_INTERNAL;
    {
      auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate);
      if (r.next == (std::numeric_limits<uint64_t>::max)()) return MGPU_VIEW_WOULD_BLOCK;
      for (size_t i = 0; i < maxOwners; ++i) if (!r.states[i]) {
        s->slot = i; s->id = r.next++; r.states[i] = s; rooted = true; break;
      }
    }
    if (!rooted) return MGPU_VIEW_WOULD_BLOCK;
    HINSTANCE module = nullptr;
    if (registerClass(module)) {
      CreateWindowExW(0, windowClass, L"", 0, 0, 0, 0, 0, HWND_MESSAGE, nullptr, module, s.get());
    }
    if (!s->window) {
      auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate);
      r.states[s->slot].reset(); return MGPU_VIEW_INTERNAL;
    }
    *details = info(*s); *out = box.release(); return MGPU_VIEW_OK;
  } catch (...) {
    if (rooted) {
      // Unknown platform setup stays owned, including any created HWND.
      std::lock_guard<std::mutex> lock(s->gate);
      s->phase = MGPU_VIEW_OWNER_CLOSE_FAILED; s->result = MGPU_VIEW_QUARANTINED;
      *details = info(*s); *out = box.release(); return MGPU_VIEW_QUARANTINED;
    }
    return MGPU_VIEW_INTERNAL;
  }
}
extern "C" int MGPUViewBeginPrivateOwnerClose(MGPUViewPrivateOwnerClose *handle,
    MGPUViewTrustedHost *host, MGPUViewPrivateDispatcher *dispatcher) {
  if (!handle || (!host && dispatcher)) return MGPU_VIEW_INVALID;
  try {
    auto s = handle->state;
    auto h = minigpu_view::private_preview::RetainPrivateHost(host);
    auto d = minigpu_view::private_dispatch::RetainPrivateDispatcher(dispatcher);
    std::lock_guard<std::mutex> lock(s->gate);
    if (s->phase == MGPU_VIEW_OWNER_CLOSE_FAILED) return MGPU_VIEW_QUARANTINED;
    if (s->phase != MGPU_VIEW_OWNER_CLOSE_READY) return MGPU_VIEW_RETIRED;
    const auto reserved = minigpu_view::private_dispatch::ReservePrivateOwnerPair(
        h, d, s->association, s->id);
    if (reserved != MGPU_VIEW_OK) return reserved;
    s->host = std::move(h); s->dispatcher = std::move(d);
    s->phase = MGPU_VIEW_OWNER_CLOSE_QUEUED;
    if (!s->window || !PostMessageW(s->window, wakeMessage, 0, 0)) {
      s->phase = MGPU_VIEW_OWNER_CLOSE_FAILED; s->result = MGPU_VIEW_QUARANTINED;
      SetEvent(s->wake); return MGPU_VIEW_QUARANTINED;
    }
    return MGPU_VIEW_OK;
  } catch (...) { return MGPU_VIEW_INTERNAL; }
}
extern "C" int MGPUViewCollectPrivateOwnerClose(MGPUViewPrivateOwnerClose *handle,
    MGPUViewPrivateOwnerCloseReceipt *out, uint32_t size) {
  if (!out || size < sizeof(*out)) return MGPU_VIEW_INVALID;
  *out = {};
  if (!handle) return MGPU_VIEW_INVALID;
  try {
    auto s = handle->state; std::lock_guard<std::mutex> lock(s->gate);
    if (s->phase == MGPU_VIEW_OWNER_CLOSE_FAILED) return MGPU_VIEW_QUARANTINED;
    if (s->phase != MGPU_VIEW_OWNER_CLOSE_COMPLETE || s->window || s->executing)
      return MGPU_VIEW_WOULD_BLOCK;
    *out = {sizeof(*out), 1, MGPU_VIEW_OK, 0, s->id, s->generation,
            s->host ? 1u : 0u, s->dispatcher ? 1u : 0u};
    return MGPU_VIEW_OK;
  } catch (...) { return MGPU_VIEW_INTERNAL; }
}
extern "C" int MGPUViewReleasePrivateOwnerClose(MGPUViewPrivateOwnerClose *handle) {
  if (!handle) return MGPU_VIEW_INVALID;
  try {
    auto s = handle->state;
    {
      std::lock_guard<std::mutex> lock(s->gate);
      if (s->phase != MGPU_VIEW_OWNER_CLOSE_COMPLETE || s->window || s->executing)
        return MGPU_VIEW_WOULD_BLOCK;
    }
    { auto &r = registry(); std::lock_guard<std::mutex> lock(r.gate);
      if (r.states[s->slot] != s) return MGPU_VIEW_INTERNAL;
      r.states[s->slot].reset(); }
    delete handle; return MGPU_VIEW_OK;
  } catch (...) { return MGPU_VIEW_INTERNAL; }
}

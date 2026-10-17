// Native contract gate for the external pool's producer handoff and output
// leases, with an in-process producer device on the pool's adapter: a canceled
// or timed-out frame hands key 0 back even after the producer released key 1,
// a producer that kept key 0 keeps it, and a second AcquireOutput of one lease
// reports CONFLICT. Also prints the producer-signal-to-ready latency.
#include "../include/minigpu.h"
#include "../include/minigpu_external_pool.h"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <d3d11_4.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
using Microsoft::WRL::ComPtr;
namespace {
constexpr unsigned width = 17, height = 9;
void require(bool yes, const char *message) {
  if (!yes) {
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(1);
  }
}
template <typename Predicate>
void until(Predicate predicate, const char *message) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
  while (!predicate() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  require(predicate(), message);
}
std::atomic<bool> initialized{false};
void ready() { initialized.store(true); }
struct Events {
  MGPUExternalPool *pool;
  std::vector<MGPUExternalPoolEvent> seen;
  void drain() {
    MGPUExternalPoolEvent e{};
    while (mgpuExternalPoolPoll(pool, &e, sizeof(e)) == 0) {
      require(e.kind != MGPU_EXTERNAL_POOL_FAULT, "no pool fault");
      seen.push_back(e);
    }
  }
  MGPUExternalPoolEvent await(uint64_t token, unsigned kind, int expected) {
    MGPUExternalPoolEvent found{};
    until(
        [&] {
          drain();
          for (auto &e : seen)
            if (e.lease_token == token && e.kind == kind) {
              found = e;
              return true;
            }
          return false;
        },
        "pool event");
    require(found.result == expected, "exact event result");
    return found;
  }
};
MGPUExternalPoolSubmission frame(uint64_t source, uint64_t fence, uint64_t seq,
                                 uint64_t deadline) {
  MGPUExternalPoolSubmission f{};
  f.struct_size = sizeof(f);
  f.version = 1;
  f.texture_id = source;
  f.producer_fence_id = fence;
  f.producer_fence_value = f.media_sequence = f.lease_token = seq;
  f.deadline_micros = mgpuExternalPoolNowMicros() + deadline;
  return f;
}
} // namespace
int main() {
  auto *context = mgpuCreateContextHandle(nullptr);
  require(context, "owned receiving context");
  mgpuContextInitializeAsync(context, ready);
  until([] { return initialized.load(); }, "GPU ready");
  MGPUExternalPoolCapabilities caps{};
  require(mgpuExternalPoolQueryCapabilities(context, &caps, sizeof(caps)) == 0,
          "pool capabilities");
  if (caps.backend != MGPU_EXTERNAL_POOL_BACKEND_D3D11) {
    mgpuDestroyContextHandle(context);
    std::cout << "SKIP: the external pool contract needs the D3D11 backend\n";
    return 0;
  }
  MGPUExternalPoolConfig config{};
  config.struct_size = sizeof(config);
  config.version = 1;
  config.width = width;
  config.height = height;
  config.pixel_format = MGPU_EXTERNAL_POOL_RGBA8;
  config.max_bytes = uint64_t(width) * height * 4 * 6;
  MGPUExternalPool *pool = nullptr;
  require(mgpuExternalPoolCreate(context, &config, &pool) == 0 && pool,
          "D3D11 pool");

  // The producer: its own D3D11 device on the pool's exact adapter.
  ComPtr<IDXGIFactory1> factory;
  require(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))), "DXGI factory");
  ComPtr<IDXGIAdapter1> adapter;
  for (UINT i = 0;; ++i) {
    ComPtr<IDXGIAdapter1> candidate;
    if (factory->EnumAdapters1(i, &candidate) == DXGI_ERROR_NOT_FOUND)
      break;
    DXGI_ADAPTER_DESC1 desc{};
    candidate->GetDesc1(&desc);
    if (desc.AdapterLuid.LowPart == caps.adapter_luid_low &&
        LONG(desc.AdapterLuid.HighPart) == LONG(caps.adapter_luid_high)) {
      adapter = candidate;
      break;
    }
  }
  require(adapter != nullptr, "pool adapter");
  ComPtr<ID3D11Device> base;
  ComPtr<ID3D11DeviceContext> immediate;
  ComPtr<ID3D11Device5> device;
  ComPtr<ID3D11DeviceContext4> producer;
  require(SUCCEEDED(D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN,
                                      nullptr, 0, nullptr, 0, D3D11_SDK_VERSION,
                                      &base, nullptr, &immediate)) &&
              SUCCEEDED(base.As(&device)) && SUCCEEDED(immediate.As(&producer)),
          "producer Device5/Context4");
  ComPtr<ID3D11Texture2D> textures[3];
  ComPtr<IDXGIKeyedMutex> keys[3];
  uint64_t sources[3]{};
  for (unsigned i = 0; i < 3; ++i) {
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE |
                     D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;
    ComPtr<IDXGIResource1> resource;
    HANDLE handle = nullptr;
    require(SUCCEEDED(device->CreateTexture2D(&desc, nullptr, &textures[i])) &&
                SUCCEEDED(textures[i].As(&keys[i])) &&
                SUCCEEDED(textures[i].As(&resource)) &&
                SUCCEEDED(resource->CreateSharedHandle(
                    nullptr,
                    DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                    nullptr, &handle)),
            "producer keyed-mutex source");
    require(mgpuExternalPoolRegisterTexture(pool, handle, &sources[i]) == 0 &&
                sources[i],
            "register source");
    CloseHandle(handle);
  }
  ComPtr<ID3D11Fence> fence;
  HANDLE fenceHandle = nullptr;
  uint64_t fenceId = 0;
  require(SUCCEEDED(device->CreateFence(0, D3D11_FENCE_FLAG_SHARED,
                                        IID_PPV_ARGS(&fence))) &&
              SUCCEEDED(fence->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr,
                                                  &fenceHandle)) &&
              mgpuExternalPoolRegisterFence(pool, fenceHandle,
                                            MGPU_EXTERNAL_POOL_PRODUCER_FENCE,
                                            &fenceId) == 0 &&
              fenceId,
          "producer fence");
  CloseHandle(fenceHandle);
  Events events{pool};

  // Documented handoff (key 1 released before Submit), then canceled while
  // the producer signal is still outstanding.
  require(keys[0]->AcquireSync(0, 1000) == S_OK &&
              keys[0]->ReleaseSync(1) == S_OK,
          "producer hands key 1 over");
  auto canceled = frame(sources[0], fenceId, 1, 2000000);
  require(mgpuExternalPoolSubmit(pool, &canceled) == MGPU_EXTERNAL_POOL_PENDING &&
              mgpuExternalPoolCancel(pool, 1) == 0,
          "submit and cancel a waiting frame");
  events.await(1, MGPU_EXTERNAL_POOL_SOURCE_DONE, MGPU_EXTERNAL_POOL_CANCELED);
  events.await(1, MGPU_EXTERNAL_POOL_OUTPUT_DONE, MGPU_EXTERNAL_POOL_CANCELED);
  require(keys[0]->AcquireSync(0, 1000) == S_OK &&
              keys[0]->ReleaseSync(0) == S_OK,
          "canceled frame hands key 0 back after key 1 was released");
  std::cout << "PASS canceled waiting frame returns key 0\n";

  // Same handoff, deadline passes before the producer signal.
  require(keys[1]->AcquireSync(0, 1000) == S_OK &&
              keys[1]->ReleaseSync(1) == S_OK,
          "producer hands key 1 over");
  auto late = frame(sources[1], fenceId, 2, 30000);
  require(mgpuExternalPoolSubmit(pool, &late) == MGPU_EXTERNAL_POOL_PENDING,
          "submit a frame that times out");
  events.await(2, MGPU_EXTERNAL_POOL_SOURCE_DONE, MGPU_EXTERNAL_POOL_TIMEOUT);
  events.await(2, MGPU_EXTERNAL_POOL_OUTPUT_DONE, MGPU_EXTERNAL_POOL_TIMEOUT);
  require(keys[1]->AcquireSync(0, 1000) == S_OK &&
              keys[1]->ReleaseSync(0) == S_OK,
          "timed-out frame hands key 0 back after key 1 was released");
  std::cout << "PASS timed-out waiting frame returns key 0\n";

  // A producer that never released key 0 keeps it through the cancel.
  require(keys[2]->AcquireSync(0, 1000) == S_OK, "producer holds key 0");
  auto held = frame(sources[2], fenceId, 3, 2000000);
  require(mgpuExternalPoolSubmit(pool, &held) == MGPU_EXTERNAL_POOL_PENDING &&
              mgpuExternalPoolCancel(pool, 3) == 0,
          "submit and cancel a held frame");
  events.await(3, MGPU_EXTERNAL_POOL_SOURCE_DONE, MGPU_EXTERNAL_POOL_CANCELED);
  require(keys[2]->ReleaseSync(0) == S_OK,
          "producer that kept key 0 still owns it");
  std::cout << "PASS held key 0 stays with the producer\n";

  // A delivered frame: signal after the coordinator went idle, then lease it.
  require(keys[0]->AcquireSync(0, 1000) == S_OK &&
              keys[0]->ReleaseSync(1) == S_OK,
          "producer hands key 1 over");
  auto delivered = frame(sources[0], fenceId, 4, 4000000);
  require(mgpuExternalPoolSubmit(pool, &delivered) ==
              MGPU_EXTERNAL_POOL_PENDING,
          "submit a delivered frame");
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  const auto signaled = std::chrono::steady_clock::now();
  require(SUCCEEDED(producer->Signal(fence.Get(), 4)), "producer signal");
  producer->Flush();
  events.await(4, MGPU_EXTERNAL_POOL_OUTPUT_READY, 0);
  const double readyMs = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - signaled)
                             .count();
  MGPUExternalPoolOutput out{}, again{};
  require(mgpuExternalPoolAcquireOutput(pool, 4, &out, sizeof(out)) == 0 &&
              out.consumer_id,
          "acquire the delivered output");
  require(mgpuExternalPoolAcquireOutput(pool, 4, &again, sizeof(again)) ==
                  MGPU_EXTERNAL_POOL_CONFLICT &&
              !again.consumer_id,
          "a second acquire of one lease reports CONFLICT");
  require(mgpuExternalPoolReleaseOutputConsumer(pool, 4, out.consumer_id, 0,
                                                0) == MGPU_EXTERNAL_POOL_PENDING,
          "release the Dawn consumer");
  events.await(4, MGPU_EXTERNAL_POOL_SOURCE_DONE, 0);
  events.await(4, MGPU_EXTERNAL_POOL_OUTPUT_DONE, 0);
  require(keys[0]->AcquireSync(0, 1000) == S_OK &&
              keys[0]->ReleaseSync(0) == S_OK,
          "delivered frame returns key 0");
  std::cout << "PASS second AcquireOutput reports CONFLICT\n";
  std::cout << "OBSERVED: producer signal to OUTPUT_READY " << readyMs
            << " ms after an idle wait\n";

  events.drain();
  require(mgpuExternalPoolRelease(pool) == 0, "pool releases");
  mgpuDestroyContextHandle(context);
  std::cout << "External pool contract test passed.\n";
  return 0;
}
#else
int main() {
  std::cout << "Windows integration gate only\n";
  return 0;
}
#endif

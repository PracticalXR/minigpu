// Native, real separate-process gate. GPU-generated source pixels are read back
// only by this test's sampler to verify contents; the pool has no readback
// path.
#include "../include/minigpu.h"
#include "../include/minigpu_external_pool.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <set>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <d3d11_4.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
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
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  require(predicate(), message);
}
void writeExact(HANDLE pipe, const void *data, DWORD size) {
  DWORD written = 0;
  require(WriteFile(pipe, data, size, &written, nullptr) && written == size,
          "pipe write");
}
void readExact(HANDLE pipe, void *data, DWORD size) {
  DWORD read = 0;
  require(ReadFile(pipe, data, size, &read, nullptr) && read == size,
          "pipe read");
}
void createDevice(LUID luid, ComPtr<ID3D11Device5> &device,
                  ComPtr<ID3D11DeviceContext4> &context) {
  ComPtr<IDXGIFactory1> factory;
  require(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))),
          "DXGI factory");
  ComPtr<IDXGIAdapter1> chosen;
  for (UINT i = 0;; ++i) {
    ComPtr<IDXGIAdapter1> adapter;
    if (factory->EnumAdapters1(i, &adapter) == DXGI_ERROR_NOT_FOUND)
      break;
    DXGI_ADAPTER_DESC1 desc{};
    adapter->GetDesc1(&desc);
    if (desc.AdapterLuid.LowPart == luid.LowPart &&
        desc.AdapterLuid.HighPart == luid.HighPart) {
      chosen = adapter;
      break;
    }
  }
  require(chosen != nullptr, "exact adapter LUID");
  ComPtr<ID3D11Device> base;
  ComPtr<ID3D11DeviceContext> immediate;
  require(SUCCEEDED(D3D11CreateDevice(chosen.Get(), D3D_DRIVER_TYPE_UNKNOWN,
                                      nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                                      nullptr, 0, D3D11_SDK_VERSION, &base,
                                      nullptr, &immediate)) &&
              SUCCEEDED(base.As(&device)) && SUCCEEDED(immediate.As(&context)),
          "Device5/Context4");
}
struct Command {
  uint32_t kind, index;
  uint64_t value;
};
struct Resources {
  uint64_t textures[3], fence, wrongTexture;
};
int producer(bool supplied = false) {
  HANDLE input = GetStdHandle(STD_INPUT_HANDLE),
         output = GetStdHandle(STD_OUTPUT_HANDLE);
  LUID luid{};
  readExact(input, &luid, sizeof(luid));
  ComPtr<ID3D11Device5> device;
  ComPtr<ID3D11DeviceContext4> context;
  createDevice(luid, device, context);
  std::array<ComPtr<ID3D11Texture2D>, 4> textures;
  std::array<ComPtr<IDXGIKeyedMutex>, 3> mutexes;
  std::array<ComPtr<ID3D11RenderTargetView>, 3> views;
  std::array<HANDLE, 5> handles{};
  Resources provided{};
  if (supplied)
    readExact(input, &provided, sizeof(provided));
  for (unsigned i = 0; i < 4; ++i) {
    if (supplied && i == 3)
      continue;
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = width + (i == 3);
    desc.Height = height;
    desc.MipLevels = desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE |
                     D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;
    if (supplied) {
      handles[i] = reinterpret_cast<HANDLE>(uintptr_t(provided.textures[i]));
      require(SUCCEEDED(device->OpenSharedResource1(
                  handles[i], IID_PPV_ARGS(&textures[i]))),
              "producer opens host-issued target in its own namespace");
    } else {
      require(SUCCEEDED(device->CreateTexture2D(&desc, nullptr, &textures[i])),
              "producer NT texture");
      ComPtr<IDXGIResource1> resource;
      require(SUCCEEDED(textures[i].As(&resource)), "texture resource");
      require(SUCCEEDED(resource->CreateSharedHandle(
                  nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                  nullptr, &handles[i])),
              "texture handle");
    }
    if (i < 3) {
      require(SUCCEEDED(textures[i].As(&mutexes[i])) &&
                  SUCCEEDED(device->CreateRenderTargetView(textures[i].Get(),
                                                           nullptr, &views[i])),
              "producer mutex/view");
    }
  }
  ComPtr<ID3D11Fence> fence;
  if (supplied) {
    handles[4] = reinterpret_cast<HANDLE>(uintptr_t(provided.fence));
    require(SUCCEEDED(device->OpenSharedFence(handles[4], IID_PPV_ARGS(&fence))),
            "producer opens host-issued fence");
  } else {
    require(SUCCEEDED(device->CreateFence(0, D3D11_FENCE_FLAG_SHARED,
                                        IID_PPV_ARGS(&fence))) &&
              SUCCEEDED(fence->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr,
                                                  &handles[4])),
            "producer shared fence");
  }
  if (supplied) {
    // Qualification-only readback on the foreign device before any app draw.
    // One staging texture verifies every pixel in all three exported sources.
    D3D11_TEXTURE2D_DESC desc{};
    textures[0]->GetDesc(&desc);
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = desc.MiscFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> staging;
    require(SUCCEEDED(device->CreateTexture2D(&desc, nullptr, &staging)),
            "initial-content staging texture");
    for (unsigned i = 0; i < 3; ++i) {
      require(mutexes[i]->AcquireSync(0, 1000) == S_OK,
              "initial source ownership on foreign device");
      context->CopyResource(staging.Get(), textures[i].Get());
      context->Flush();
      D3D11_MAPPED_SUBRESOURCE mapped{};
      const auto deadline = std::chrono::steady_clock::now() +
                            std::chrono::seconds(5);
      HRESULT result;
      do {
        result = context->Map(staging.Get(), 0, D3D11_MAP_READ,
                              D3D11_MAP_FLAG_DO_NOT_WAIT, &mapped);
        if (result != DXGI_ERROR_WAS_STILL_DRAWING)
          break;
        require(std::chrono::steady_clock::now() < deadline,
                "initial-content copy completes within deadline");
        Sleep(1);
      } while (true);
      require(result == S_OK && mapped.pData && mapped.RowPitch >= width * 4,
              "initial-content readback completes");
      for (unsigned y = 0; y < height; ++y) {
        const auto *row = static_cast<const uint8_t *>(mapped.pData) +
                          size_t(y) * mapped.RowPitch;
        for (unsigned x = 0; x < width * 4; ++x)
          require(row[x] == 0, "all issued source pixels initially zero");
      }
      context->Unmap(staging.Get(), 0);
      require(mutexes[i]->ReleaseSync(0) == S_OK,
              "initial source inspection preserves producer key");
    }
  }
  Resources reply{};
  for (unsigned i = 0; i < 3; ++i)
    reply.textures[i] = reinterpret_cast<uintptr_t>(handles[i]);
  reply.wrongTexture = reinterpret_cast<uintptr_t>(handles[3]);
  reply.fence = reinterpret_cast<uintptr_t>(handles[4]);
  writeExact(output, &reply, sizeof(reply));
  for (;;) {
    Command cmd{};
    readExact(input, &cmd, sizeof(cmd));
    if (!cmd.kind)
      break;
    require(cmd.index < 3, "producer slot");
    if (cmd.kind == 3) {
      require(mutexes[cmd.index]->ReleaseSync(0) == S_OK,
              "release deliberately held mutex");
    } else {
      require(mutexes[cmd.index]->AcquireSync(0, 1000) == S_OK,
              "producer key zero available after SOURCE_DONE");
      float color[] = {float(cmd.value) / 255.0f, 0.25f, 0.5f, 1.0f};
      context->ClearRenderTargetView(views[cmd.index].Get(), color);
      require(SUCCEEDED(context->Signal(fence.Get(), cmd.value)),
              "producer signal");
      context->Flush();
      if (cmd.kind == 1)
        require(mutexes[cmd.index]->ReleaseSync(1) == S_OK,
                "producer key one handoff");
    }
    uint32_t ack = 1;
    writeExact(output, &ack, sizeof(ack));
  }
  for (auto handle : handles)
    if (handle)
      CloseHandle(handle);
  return 0;
}
class Child {
public:
  explicit Child(LUID luid, const MGPUExternalPoolProducer *issued = nullptr) {
    SECURITY_ATTRIBUTES attributes{sizeof(attributes), nullptr, TRUE};
    HANDLE childRead, childWrite;
    require(CreatePipe(&childRead, &write, &attributes, 0) &&
                CreatePipe(&read, &childWrite, &attributes, 0),
            "create producer pipes");
    require(SetHandleInformation(write, HANDLE_FLAG_INHERIT, 0) &&
                SetHandleInformation(read, HANDLE_FLAG_INHERIT, 0),
            "noninheritable parent handles");
    wchar_t exe[MAX_PATH];
    require(GetModuleFileNameW(nullptr, exe, MAX_PATH), "own executable");
    std::wstring cmd = L"\"" + std::wstring(exe) +
                       (issued ? L"\" --issued-producer" : L"\" --producer");
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = childRead;
    startup.hStdOutput = childWrite;
    startup.hStdError = childWrite;
    require(CreateProcessW(exe, cmd.data(), nullptr, nullptr, TRUE,
                           CREATE_NO_WINDOW, nullptr, nullptr, &startup,
                           &process),
            "separate producer process");
    CloseHandle(childRead);
    CloseHandle(childWrite);
    writeExact(write, &luid, sizeof(luid));
    if (issued) {
      auto provide = [&](void *local) {
        HANDLE remote = nullptr;
        require(DuplicateHandle(GetCurrentProcess(), local, process.hProcess,
                                &remote, 0, FALSE, DUPLICATE_SAME_ACCESS),
                "duplicate into exact owned producer process namespace");
        return uint64_t(reinterpret_cast<uintptr_t>(remote));
      };
      Resources provided{};
      for (unsigned i = 0; i < 3; ++i)
        provided.textures[i] = provide(issued->nt_texture_handles[i]);
      provided.fence = provide(issued->nt_producer_fence_handle);
      writeExact(write, &provided, sizeof(provided));
    }
    readExact(read, &resources, sizeof(resources));
  }
  HANDLE duplicate(uint64_t value) {
    HANDLE local = nullptr;
    require(DuplicateHandle(
                process.hProcess, reinterpret_cast<HANDLE>(uintptr_t(value)),
                GetCurrentProcess(), &local, 0, FALSE, DUPLICATE_SAME_ACCESS),
            "actual process resource duplicate");
    return local;
  }
  void command(unsigned kind, unsigned index, uint64_t value) {
    Command cmd{kind, index, value};
    writeExact(write, &cmd, sizeof(cmd));
    uint32_t ack = 0;
    readExact(read, &ack, sizeof(ack));
    require(ack == 1, "producer ack");
  }
  void stop() {
    if (!process.hProcess)
      return;
    Command cmd{};
    writeExact(write, &cmd, sizeof(cmd));
    require(WaitForSingleObject(process.hProcess, 5000) == WAIT_OBJECT_0,
            "producer exits");
    DWORD code = 1;
    GetExitCodeProcess(process.hProcess, &code);
    require(code == 0, "producer succeeds");
    CloseHandle(process.hProcess);
    CloseHandle(process.hThread);
    process = {};
    CloseHandle(read);
    CloseHandle(write);
    read = write = nullptr;
  }
  ~Child() { stop(); }
  Resources resources{};
  HANDLE read = nullptr, write = nullptr;
  PROCESS_INFORMATION process{};
};
std::atomic<bool> initialized{false};
std::atomic<bool> sampleReady{false};
void sampled() { sampleReady.store(true); }
void ready() { initialized.store(true); }
std::atomic<bool> workerPaused{false}, resumeWorker{false},
    timelineBlocked{false};
ComPtr<ID3D11DeviceContext4> receivingContext;
ComPtr<ID3D11Fence> receivingGate;
void pauseWorker() {
  workerPaused.store(true);
  until([] { return resumeWorker.load(); }, "test releases GPU worker gate");
}
void blockTimeline() {
  // Test-only trusted host fence: serialized on the receiving worker. Foreign
  // readiness still never inserts a wait onto that queue in the pool itself.
  require(SUCCEEDED(receivingContext->Wait(receivingGate.Get(), 1)),
          "test trusted host GPU wait");
  receivingContext->Flush();
  timelineBlocked.store(true);
}
constexpr const char *sampleKernel = R"WGSL(
@group(0) @binding(0) var frame: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> pixels: array<u32>;
@compute @workgroup_size(1) fn main() {
  let rgba = vec4<u32>(textureLoad(frame, vec2<i32>(0,0), 0) * 255.0 + 0.5);
  pixels[0] = rgba.r | (rgba.g << 8u) | (rgba.b << 16u) | (rgba.a << 24u);
})WGSL";
class Events {
public:
  explicit Events(MGPUExternalPool *value) : pool(value) {}
  void drain() {
    MGPUExternalPoolEvent e{};
    int result;
    while ((result = mgpuExternalPoolPoll(pool, &e, sizeof(e))) == 0) {
      require(allowFault || e.kind != MGPU_EXTERNAL_POOL_FAULT,
              "no unexpected pool fault");
      seen.push_back(e);
    }
    require(result == MGPU_EXTERNAL_POOL_EMPTY, "bounded event polling");
  }
  MGPUExternalPoolEvent await(uint64_t token, unsigned kind, int expected = 0) {
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
        "completion event");
    require(found.result == expected, "exact event result");
    return found;
  }
  MGPUExternalPool *pool;
  bool allowFault = false;
  std::vector<MGPUExternalPoolEvent> seen;
};
MGPUExternalPoolStats stats(MGPUExternalPool *pool) {
  MGPUExternalPoolStats value{};
  require(mgpuExternalPoolGetStats(pool, &value, sizeof(value)) == 0,
          "pool stats");
  return value;
}
MGPUExternalPoolSubmission frame(uint64_t source, uint64_t fence,
                                 uint64_t value, uint64_t seq,
                                 uint64_t deadline = 500000) {
  MGPUExternalPoolSubmission f{};
  f.struct_size = sizeof(f);
  f.version = 1;
  f.texture_id = source;
  f.producer_fence_id = fence;
  f.producer_fence_value = value;
  f.media_sequence = f.lease_token = seq;
  f.deadline_micros = mgpuExternalPoolNowMicros() + deadline;
  return f;
}
MGPUExternalPoolOutput acquire(MGPUExternalPool *pool, uint64_t token) {
  MGPUExternalPoolOutput out{};
  require(mgpuExternalPoolAcquireOutput(pool, token, &out, sizeof(out)) == 0 &&
              out.wgpu_texture_view && out.consumer_id &&
              out.context_generation && out.device_generation,
          "acquire retained output");
  return out;
}
void release(MGPUExternalPool *pool, MGPUExternalPoolOutput out,
             Events &events) {
  require(mgpuExternalPoolReleaseOutputConsumer(pool, out.lease_token,
                                                out.consumer_id, 0, 0) ==
              MGPU_EXTERNAL_POOL_PENDING,
          "Dawn completion queued");
  events.await(out.lease_token, MGPU_EXTERNAL_POOL_OUTPUT_DONE);
}
} // namespace
int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--producer")
    return producer();
  if (argc == 2 && std::string(argv[1]) == "--issued-producer")
    return producer(true);
  const bool issued = argc == 2 && std::string(argv[1]) == "--issued";
  const unsigned expectedAllocations = issued ? 6 : 3;
  auto createPool = issued ? mgpuExternalPoolCreateIssued : mgpuExternalPoolCreate;
  const bool freshIdentityProbe =
      argc == 2 && std::string(argv[1]) == "--fresh-identity-probe";
  auto *context = mgpuCreateContextHandle(nullptr);
  require(context, "owned receiving context");
  MGPUExternalPoolConfig config{};
  MGPUExternalPoolLimits baseline{}, limits{};
  require(mgpuExternalPoolGetLimits(&baseline, sizeof(baseline)) == 0 &&
              baseline.max_pools == 16 && baseline.output_slots == 3 &&
              baseline.pixel_format_mask == 3,
          "query bounded native limits/formats");
  config.struct_size = sizeof(config);
  config.version = 1;
  config.width = width;
  config.height = height;
  config.pixel_format = MGPU_EXTERNAL_POOL_RGBA8;
  config.max_bytes = uint64_t(width) * height * 4 * 6;
  MGPUExternalPool *pool = reinterpret_cast<MGPUExternalPool *>(uintptr_t(1));
  require(createPool(context, &config, &pool) ==
                  MGPU_CONTEXT_LEASE_NOT_READY &&
              !pool,
          "pool cannot initialize receiving device");
  mgpuContextInitializeAsync(context, ready);
  until([] { return initialized.load(); }, "GPU ready");
  MGPUExternalPoolCapabilities caps{};
  require(mgpuExternalPoolQueryCapabilities(context, &caps, sizeof(caps)) == 0,
          "actual facilities");
  if (caps.backend != MGPU_EXTERNAL_POOL_BACKEND_D3D11) {
    require(createPool(context, &config, &pool) ==
                    MGPU_EXTERNAL_POOL_UNSUPPORTED &&
                !pool,
            "unsupported backend refuses pool without casts/fallback");
    mgpuDestroyContextHandle(context);
    std::cout << "PASS: explicit unsupported backend\n";
    return 0;
  }
  require(caps.refusal == MGPU_EXTERNAL_POOL_FACILITIES_READY &&
              createPool(context, &config, &pool) == 0 && pool,
          "real D3D11 host output pool");
  MGPUExternalPoolProducer provided{};
  if (issued) {
    auto beforeExport = frame(1, 4, 1, 1);
    require(mgpuExternalPoolSubmit(pool, &beforeExport) ==
                MGPU_CONTEXT_LEASE_RETIRED,
            "issued source is unavailable before exporter admission");
    require(mgpuExternalPoolAcquireProducer(pool, &provided, sizeof(provided)) == 0 &&
                provided.slot_count == 3 && provided.export_lease &&
                provided.context_generation == caps.context_generation &&
                provided.device_generation == caps.device_generation,
            "exact issued pool and exporter lease");
    MGPUExternalPoolProducer duplicate{};
    require(mgpuExternalPoolAcquireProducer(pool, &duplicate, sizeof(duplicate)) ==
                    MGPU_EXTERNAL_POOL_CONFLICT && !duplicate.export_lease &&
                mgpuExternalPoolReleaseProducerAfterPeerRelease(
                    pool, provided.export_lease + 1) == MGPU_CONTEXT_LEASE_INVALID &&
                mgpuExternalPoolRelease(pool) == MGPU_CONTEXT_LEASE_WOULD_BLOCK,
            "one exporter, exact retirement identity and retained outbound quota");
  } else {
    require(mgpuExternalPoolAcquireProducer(pool, &provided, sizeof(provided)) ==
                MGPU_EXTERNAL_POOL_UNSUPPORTED,
            "legacy pool cannot supply issued producer access");
  }
  Child child({caps.adapter_luid_low, caps.adapter_luid_high},
              issued ? &provided : nullptr);
  require(mgpuExternalPoolGetLimits(&limits, sizeof(limits)) == 0 &&
              limits.live_pools == baseline.live_pools + 1 &&
              limits.reserved_bytes ==
                  baseline.reserved_bytes + config.max_bytes,
          "global quota includes configured pool reservation");
  uint64_t bad = 123;
  HANDLE h = nullptr;
  if (!issued) {
    h = child.duplicate(child.resources.wrongTexture);
    require(mgpuExternalPoolRegisterTexture(pool, h, &bad) ==
                    MGPU_CONTEXT_LEASE_INVALID &&
                !bad,
            "actual descriptor mismatch rejected");
    CloseHandle(h);
    h = child.duplicate(child.resources.fence);
    require(mgpuExternalPoolRegisterTexture(pool, h, &bad) ==
                    MGPU_CONTEXT_LEASE_INVALID &&
                !bad,
            "fence is not a texture");
    CloseHandle(h);
    h = child.duplicate(child.resources.textures[0]);
    require(mgpuExternalPoolRegisterFence(pool, h,
                                         MGPU_EXTERNAL_POOL_PRODUCER_FENCE,
                                         &bad) == MGPU_CONTEXT_LEASE_INVALID &&
                !bad,
            "texture is not a fence");
    CloseHandle(h);
  }
  uint64_t sources[3]{};
  for (unsigned i = 0; i < 3; ++i) {
    h = child.duplicate(child.resources.textures[i]);
    if (issued) {
      sources[i] = provided.texture_ids[i];
      require(sources[i] &&
                  mgpuExternalPoolRegisterTexture(pool, h, &bad) ==
                      MGPU_EXTERNAL_POOL_CONFLICT && !bad &&
                  mgpuExternalPoolUnregisterTexture(pool, sources[i]) ==
                      MGPU_EXTERNAL_POOL_CONFLICT,
              "issued source identity cannot be replaced or unregistered");
    } else {
      require(mgpuExternalPoolRegisterTexture(pool, h, &sources[i]) == 0 &&
                  sources[i],
              "cached NT source registration");
      if (i == 0) {
        HANDLE alias = child.duplicate(child.resources.textures[i]);
        require(alias != h &&
                    mgpuExternalPoolRegisterTexture(pool, alias, &bad) ==
                        MGPU_EXTERNAL_POOL_CONFLICT && !bad,
                "different NT handle cannot bypass source identity");
        CloseHandle(alias);
      } else if (i == 2) {
        auto retiredId = sources[i];
        require(mgpuExternalPoolUnregisterTexture(pool, retiredId) == 0 &&
                    mgpuExternalPoolRegisterTexture(pool, h, &sources[i]) == 0 &&
                    sources[i] != retiredId,
                "safe unregister releases the exact allocation identity");
      }
    }
    CloseHandle(h);
  }
  h = child.duplicate(child.resources.textures[0]);
  require(mgpuExternalPoolRegisterTexture(pool, h, &bad) ==
                  (issued ? MGPU_EXTERNAL_POOL_CONFLICT :
                            MGPU_CONTEXT_LEASE_WOULD_BLOCK) &&
              !bad,
          "native source budget enforced");
  CloseHandle(h);
  MGPUExternalPool *aliasPool = nullptr;
  require(mgpuExternalPoolCreate(context, &config, &aliasPool) == 0 && aliasPool,
          "second live pool for alias admission");
  HANDLE aliasTexture = child.duplicate(child.resources.textures[1]);
  require(mgpuExternalPoolRegisterTexture(aliasPool, aliasTexture, &bad) ==
                  MGPU_EXTERNAL_POOL_CONFLICT && !bad,
          "a second pool cannot register a still-owned allocation");
  uint64_t producerFence = 0;
  h = child.duplicate(child.resources.fence);
  if (issued) {
    producerFence = provided.producer_fence_id;
    require(mgpuExternalPoolRegisterFence(
                pool, h, MGPU_EXTERNAL_POOL_PRODUCER_FENCE, &bad) ==
                    MGPU_EXTERNAL_POOL_CONFLICT && !bad &&
                mgpuExternalPoolUnregisterFence(pool, producerFence) ==
                    MGPU_EXTERNAL_POOL_CONFLICT,
            "issued producer fence cannot be replaced or unregistered");
  } else {
    require(mgpuExternalPoolRegisterFence(
                pool, h, MGPU_EXTERNAL_POOL_PRODUCER_FENCE, &producerFence) == 0,
            "typed producer fence registration");
  }
  require(mgpuExternalPoolRegisterFence(
              pool, h, MGPU_EXTERNAL_POOL_PRODUCER_FENCE, &bad) ==
                  MGPU_EXTERNAL_POOL_CONFLICT && !bad &&
              mgpuExternalPoolRegisterFence(
                  pool, h, MGPU_EXTERNAL_POOL_HOST_CONSUMER_FENCE, &bad) ==
                  MGPU_EXTERNAL_POOL_CONFLICT && !bad &&
              mgpuExternalPoolRegisterFence(
                  aliasPool, h, MGPU_EXTERNAL_POOL_HOST_CONSUMER_FENCE, &bad) ==
                  MGPU_EXTERNAL_POOL_CONFLICT && !bad,
          "producer fence aliases cannot reset values or become host fences");
  CloseHandle(h);
  ComPtr<ID3D11Device5> hostDevice;
  ComPtr<ID3D11DeviceContext4> hostContext;
  createDevice({caps.adapter_luid_low, caps.adapter_luid_high}, hostDevice,
               hostContext);
  ComPtr<ID3D11Fence> encoderFence;
  require(SUCCEEDED(hostDevice->CreateFence(0, D3D11_FENCE_FLAG_SHARED,
                                            IID_PPV_ARGS(&encoderFence))) &&
              SUCCEEDED(encoderFence->CreateSharedHandle(nullptr, GENERIC_ALL,
                                                         nullptr, &h)),
          "host encoder fence");
  uint64_t hostFence = 0;
  require(mgpuExternalPoolRegisterFence(
              pool, h, MGPU_EXTERNAL_POOL_HOST_CONSUMER_FENCE, &hostFence) == 0,
          "separate host completion registration");
  CloseHandle(h);
  // Keep the invasive fresh-sharing qualification separate from the retained
  // pixel/lifetime gate. These unverified aliases are never submitted for GPU
  // reads, promoted to an authenticated resource or exposed to an application.
  if (freshIdentityProbe) {
    require(SUCCEEDED(encoderFence->CreateSharedHandle(nullptr, GENERIC_ALL,
                                                       nullptr, &h)),
            "second NT handle for the same native fence");
    auto freshFence = mgpuExternalPoolRegisterFence(
        pool, h, MGPU_EXTERNAL_POOL_PRODUCER_FENCE, &bad);
    if (freshFence == 0) {
      require(bad && mgpuExternalPoolUnregisterFence(pool, bad) == 0,
              "retire unsubmitted fresh-handle fence probe");
      std::cout << "LIMIT: fresh shared fence handles are distinct kernel objects; "
                   "canonical GPU identity remains unqualified\n";
    } else {
      require(freshFence == MGPU_EXTERNAL_POOL_CONFLICT && !bad,
              "equivalent kernel fence identities refuse");
    }
    CloseHandle(h);
    ComPtr<ID3D11Texture2D> reopenedTexture;
    ComPtr<IDXGIResource1> reopenedResource;
    require(SUCCEEDED(hostDevice->OpenSharedResource1(
                aliasTexture, IID_PPV_ARGS(&reopenedTexture))) &&
                SUCCEEDED(reopenedTexture.As(&reopenedResource)),
            "independent native wrapper for the source allocation");
    if (SUCCEEDED(reopenedResource->CreateSharedHandle(
            nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
            nullptr, &h))) {
      auto freshTexture = mgpuExternalPoolRegisterTexture(aliasPool, h, &bad);
      if (freshTexture == 0) {
        require(bad && mgpuExternalPoolUnregisterTexture(aliasPool, bad) == 0,
                "retire unsubmitted reopened-texture probe");
        std::cout << "LIMIT: reopened texture minted a distinct kernel object; "
                     "canonical GPU identity remains unqualified\n";
      } else {
        require(freshTexture == MGPU_EXTERNAL_POOL_CONFLICT && !bad,
                "equivalent kernel texture identities refuse");
      }
      CloseHandle(h);
    }
    reopenedResource.Reset();
    reopenedTexture.Reset();
    CloseHandle(aliasTexture);
    require(mgpuExternalPoolRelease(aliasPool) == 0 &&
                mgpuExternalPoolRelease(pool) == 0,
            "retire cold identity probes without submitted GPU work");
    child.stop();
    mgpuDestroyContextHandle(context);
    until(
        [&] {
          return mgpuExternalPoolGetLimits(&limits, sizeof(limits)) == 0 &&
                 limits.live_pools == baseline.live_pools &&
                 limits.reserved_bytes == baseline.reserved_bytes;
        },
        "identity qualification releases its actual fixed owners");
    std::cout << "OBSERVED: cold shared-handle identity and actual owner cleanup; "
                 "no renderer/encoder/canonical-identity qualification\n";
    return 0;
  }
  Events events(pool);
  // A missing signal and a signaled-but-held foreign mutex never occupy the
  // shared GPU queue. Both time out before copy and drain without allocations.
  auto missing = frame(sources[0], producerFence, 1, 1, 30000);
  require(mgpuExternalPoolSubmit(pool, &missing) == MGPU_EXTERNAL_POOL_PENDING,
          "accept unsignaled source");
  require(mgpuExternalPoolUnregisterTexture(pool, sources[0]) ==
                  (issued ? MGPU_EXTERNAL_POOL_CONFLICT :
                            MGPU_CONTEXT_LEASE_WOULD_BLOCK) &&
              mgpuExternalPoolRelease(pool) == MGPU_CONTEXT_LEASE_WOULD_BLOCK,
          "waiting source stays owned/charged");
  events.await(1, MGPU_EXTERNAL_POOL_OUTPUT_DONE, MGPU_EXTERNAL_POOL_TIMEOUT);
  require(!stats(pool).gpu_copies, "unsignaled source never copied");
  child.command(2, 0, 2);
  auto held = frame(sources[0], producerFence, 2, 2, 30000);
  require(mgpuExternalPoolSubmit(pool, &held) == MGPU_EXTERNAL_POOL_PENDING,
          "accept signaled held source");
  events.await(2, MGPU_EXTERNAL_POOL_OUTPUT_DONE, MGPU_EXTERNAL_POOL_TIMEOUT);
  require(!stats(pool).gpu_copies, "nonnegative WAIT_TIMEOUT never admitted");
  child.command(3, 0, 0);
  auto waitingCancel = frame(sources[0], producerFence, 3, 3);
  require(mgpuExternalPoolSubmit(pool, &waitingCancel) ==
                  MGPU_EXTERNAL_POOL_PENDING &&
              mgpuExternalPoolCancel(pool, 3) == 0,
          "cancel Waiting source");
  events.await(3, MGPU_EXTERNAL_POOL_OUTPUT_DONE, MGPU_EXTERNAL_POOL_CANCELED);
  auto *gateShader = mgpuContextCreateComputeShader(context);
  require(gateShader, "test GPU-worker gate shader");
  mgpuLoadKernel(gateShader, "@compute @workgroup_size(1) fn main() {}");
  mgpuDispatchAsync(gateShader, 1, 1, 1, pauseWorker);
  until([] { return workerPaused.load(); }, "GPU-worker gate entered");
  child.command(1, 0, 4);
  auto queuedCancel = frame(sources[0], producerFence, 4, 4);
  require(mgpuExternalPoolSubmit(pool, &queuedCancel) ==
              MGPU_EXTERNAL_POOL_PENDING,
          "accept source while receiver worker gated");
  until([&] { return stats(pool).queued_copies == 1; },
        "source acquired and CopyQueued");
  require(mgpuExternalPoolCancel(pool, 4) == 0,
          "cancel CopyQueued source with held key one");
  resumeWorker.store(true);
  auto canceledSource = events.await(4, MGPU_EXTERNAL_POOL_SOURCE_DONE,
                                     MGPU_EXTERNAL_POOL_CANCELED);
  require(!canceledSource.copy_fence_value && !stats(pool).gpu_copies,
          "queued cancellation returns source without submitting GPU work");
  events.await(4, MGPU_EXTERNAL_POOL_OUTPUT_DONE, MGPU_EXTERNAL_POOL_CANCELED);
  mgpuDestroyComputeShader(gateShader);
  auto *shader = mgpuContextCreateComputeShader(context);
  auto *pixels = mgpuContextCreateBuffer(context, 4, 7);
  require(shader && pixels, "persistent sampler resources");
  mgpuLoadKernel(shader, sampleKernel);
  mgpuSetBufferFire(shader, 1, pixels);
  std::set<void *> allocations;
  uint64_t sequence = 5;
  for (; sequence < 13; ++sequence) {
    unsigned i = unsigned(sequence % 3);
    child.command(1, i, sequence);
    auto f = frame(sources[i], producerFence, sequence, sequence);
    require(mgpuExternalPoolSubmit(pool, &f) == MGPU_EXTERNAL_POOL_PENDING,
            "submit registered reusable source");
    auto sourceDone = events.await(sequence, MGPU_EXTERNAL_POOL_SOURCE_DONE);
    require(sourceDone.copy_fence_value,
            "actual copy fence before producer reuse");
    events.await(sequence, MGPU_EXTERNAL_POOL_OUTPUT_READY);
    auto out = acquire(pool, sequence);
    allocations.insert(out.d3d11_texture);
    require(mgpuExternalPoolBindOutput(pool, sequence, out.consumer_id, shader,
                                       0) == 0,
            "context-owned output binding");
    mgpuDispatch(shader, 1, 1, 1);
    uint32_t pixel = 0;
    // Queue readback after binding/dispatch on the creation worker. The direct
    // synchronous read can acquire the GPU mutex before those queued tasks and
    // return the preceding frame; its CPU completion is not dispatch completion.
    sampleReady.store(false);
    mgpuReadAsyncUint32(pixels, &pixel, 1, 0, sampled);
    until([] { return sampleReady.load(); }, "ordered sampler readback");
    if ((pixel & 255) != sequence || ((pixel >> 24) & 255) != 255)
      std::cerr << "Sample mismatch: sequence=" << sequence << " slot=" << i
                << " pixel=" << pixel << std::endl;
    require((pixel & 255) == sequence && ((pixel >> 24) & 255) == 255,
            "sampler sees exact GPU-generated source pixels");
    release(pool, out, events);
    require(mgpuExternalPoolSubmit(pool, &f) == MGPU_EXTERNAL_POOL_CONFLICT,
            "accepted token/sequence cannot replay");
  }
  require(stats(pool).host_allocations == expectedAllocations &&
              stats(pool).host_imports == 3 &&
              stats(pool).texture_count == 3 && stats(pool).gpu_copies == 8 &&
              allocations.size() <= 3,
          "constant host allocation/import counts across reuse");
  // Three claimed outputs create backpressure even though all three producers
  // have already completed. A producer fence cannot release an encoder lease.
  std::vector<MGPUExternalPoolOutput> occupied;
  uint64_t encoderConsumer = 0;
  for (unsigned i = 0; i < 3; ++i, ++sequence) {
    child.command(1, i, sequence);
    auto f = frame(sources[i], producerFence, sequence, sequence);
    require(mgpuExternalPoolSubmit(pool, &f) == MGPU_EXTERNAL_POOL_PENDING,
            "parallel host outputs");
    events.await(sequence, MGPU_EXTERNAL_POOL_SOURCE_DONE);
    events.await(sequence, MGPU_EXTERNAL_POOL_OUTPUT_READY);
    occupied.push_back(acquire(pool, sequence));
    if (!i) {
      require(mgpuExternalPoolRetainOutput(
                  pool, sequence, MGPU_EXTERNAL_POOL_NATIVE_FENCE_CONSUMER,
                  &encoderConsumer) == 0,
              "retain independent encoder consumer");
      require(mgpuExternalPoolReleaseOutputConsumer(
                  pool, sequence, encoderConsumer, producerFence, sequence) ==
                  MGPU_CONTEXT_LEASE_INVALID,
              "producer fence cannot free host output");
      require(mgpuExternalPoolReleaseOutputConsumer(
                  pool, sequence, encoderConsumer, hostFence, 1) ==
                  MGPU_EXTERNAL_POOL_PENDING,
              "encoder waits on its own fence");
    }
  }
  auto blocked = frame(sources[0], producerFence, sequence, sequence);
  require(mgpuExternalPoolSubmit(pool, &blocked) ==
                  MGPU_CONTEXT_LEASE_WOULD_BLOCK &&
              !stats(pool).pending_sources,
          "held output admission consumes no source ownership");
  require(mgpuExternalPoolUnregisterFence(pool, hostFence) ==
              MGPU_CONTEXT_LEASE_WOULD_BLOCK,
          "pending encoder fence remains registered");
  require(mgpuExternalPoolReleaseOutputConsumer(
              pool, occupied[0].lease_token, occupied[0].consumer_id, 0, 0) ==
              MGPU_EXTERNAL_POOL_PENDING,
          "release sampler while encoder holds output");
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  events.drain();
  require(stats(pool).occupied_outputs == 3,
          "sampler completion cannot recycle encoder-held output");
  require(SUCCEEDED(hostContext->Signal(encoderFence.Get(), 1)),
          "actual encoder signal");
  hostContext->Flush();
  events.await(occupied[0].lease_token, MGPU_EXTERNAL_POOL_OUTPUT_DONE);
  release(pool, occupied[1], events);
  release(pool, occupied[2], events);
  // Both profiles retain actual resources. Legacy admission remains valid after
  // producer exit; issued admission retains its exporter until the final proof.
  child.command(1, 0, sequence);
  child.command(1, 1, sequence + 1);
  if (!issued)
    child.stop();
  auto finalFrame = frame(sources[0], producerFence, sequence, sequence);
  require(mgpuExternalPoolSubmit(pool, &finalFrame) ==
              MGPU_EXTERNAL_POOL_PENDING,
          "retained source remains valid for the final copy");
  events.await(sequence, MGPU_EXTERNAL_POOL_SOURCE_DONE);
  events.await(sequence, MGPU_EXTERNAL_POOL_OUTPUT_READY);
  auto finalOut = acquire(pool, sequence);
  uint64_t consumers[3]{};
  for (auto &id : consumers)
    require(mgpuExternalPoolRetainOutput(
                pool, sequence, MGPU_EXTERNAL_POOL_DAWN_CONSUMER, &id) == 0,
            "bounded retained sampler consumers");
  require(mgpuExternalPoolRetainOutput(
              pool, sequence, MGPU_EXTERNAL_POOL_DAWN_CONSUMER, &bad) ==
                  MGPU_CONTEXT_LEASE_WOULD_BLOCK &&
              !bad,
          "consumer capacity clears output on refusal");
  ComPtr<ID3D11Device> receivingBase;
  reinterpret_cast<ID3D11Texture2D *>(finalOut.d3d11_texture)
      ->GetDevice(&receivingBase);
  ComPtr<ID3D11Device5> receivingDevice;
  ComPtr<ID3D11DeviceContext> immediate;
  require(SUCCEEDED(receivingBase.As(&receivingDevice)),
          "receiving exact native device");
  receivingBase->GetImmediateContext(&immediate);
  require(SUCCEEDED(immediate.As(&receivingContext)),
          "receiving exact native context");
  ComPtr<ID3D11Fence> hostGate;
  require(SUCCEEDED(hostDevice->CreateFence(0, D3D11_FENCE_FLAG_SHARED,
                                            IID_PPV_ARGS(&hostGate))) &&
              SUCCEEDED(hostGate->CreateSharedHandle(nullptr, GENERIC_ALL,
                                                     nullptr, &h)) &&
              SUCCEEDED(receivingDevice->OpenSharedFence(
                  h, IID_PPV_ARGS(&receivingGate))),
          "test host-owned cross-context completion gate");
  CloseHandle(h);
  require(mgpuExternalPoolBindOutput(pool, sequence, finalOut.consumer_id,
                                     shader, 0) == 0,
          "bind final live consumer before test GPU gate");
  mgpuDispatchAsync(shader, 1, 1, 1, blockTimeline);
  until([] { return timelineBlocked.load(); }, "trusted host GPU wait queued");
  auto copyCount = stats(pool).gpu_copies;
  auto uncertain =
      frame(sources[1], producerFence, sequence + 1, sequence + 1, 30000);
  require(mgpuExternalPoolSubmit(pool, &uncertain) ==
              MGPU_EXTERNAL_POOL_PENDING,
          "submitted copy behind actual host GPU fence wait");
  until([&] { return stats(pool).gpu_copies == copyCount + 1; },
        "real source copy submitted");
  require(mgpuExternalPoolCancel(pool, sequence + 1) == 0,
          "cancel real submitted copy");
  events.allowFault = true;
  events.await(0, MGPU_EXTERNAL_POOL_FAULT, MGPU_EXTERNAL_POOL_TIMEOUT);
  require(stats(pool).pending_sources == 1 &&
              stats(pool).occupied_outputs == 2 &&
              stats(pool).quarantined_outputs == 1 &&
              stats(pool).reserved_bytes == config.max_bytes &&
              mgpuExternalPoolUnregisterTexture(pool, sources[1]) ==
                  (issued ? MGPU_EXTERNAL_POOL_CONFLICT :
                            MGPU_CONTEXT_LEASE_WOULD_BLOCK) &&
              mgpuExternalPoolRelease(pool) == MGPU_CONTEXT_LEASE_WOULD_BLOCK,
          "post-submit timeout keeps source/output/resources charged");
  for (const auto &e : events.seen)
    require(e.lease_token != sequence + 1 ||
                e.kind != MGPU_EXTERNAL_POOL_SOURCE_DONE,
            "cancel/timeout cannot invent source completion");
  require(mgpuExternalPoolRetire(pool) == 0 &&
              mgpuExternalPoolBindOutput(pool, sequence, finalOut.consumer_id,
                                         shader,
                                         0) == MGPU_CONTEXT_LEASE_RETIRED,
          "retirement stops new use while retaining consumers");
  require(mgpuExternalPoolRegisterTexture(aliasPool, aliasTexture, &bad) ==
                  MGPU_EXTERNAL_POOL_CONFLICT && !bad,
          "retiring pool retains kernel identity through actual GPU completion");
  CloseHandle(aliasTexture);
  require(mgpuExternalPoolRelease(aliasPool) == 0,
          "empty alias probe pool retires its fixed resources");
  mgpuDestroyComputeShader(shader);
  mgpuDestroyBuffer(pixels);
  mgpuDestroyContextHandle(context);
  for (auto id : consumers)
    require(mgpuExternalPoolReleaseOutputConsumer(pool, sequence, id, 0, 0) ==
                MGPU_EXTERNAL_POOL_PENDING,
            "consumer drains exact queue after context retirement");
  require(mgpuExternalPoolReleaseOutputConsumer(pool, sequence,
                                                finalOut.consumer_id, 0, 0) ==
              MGPU_EXTERNAL_POOL_PENDING,
          "initial consumer retirement completion");
  require(SUCCEEDED(hostContext->Signal(hostGate.Get(), 1)),
          "actual host completion releases submitted copy");
  hostContext->Flush();
  auto drainedCopy = events.await(sequence + 1, MGPU_EXTERNAL_POOL_SOURCE_DONE,
                                  MGPU_EXTERNAL_POOL_CANCELED);
  require(drainedCopy.copy_fence_value,
          "canceled submitted copy reports actual completion fence");
  events.await(sequence + 1, MGPU_EXTERNAL_POOL_OUTPUT_DONE,
               MGPU_EXTERNAL_POOL_CANCELED);
  events.await(sequence, MGPU_EXTERNAL_POOL_OUTPUT_DONE,
               MGPU_EXTERNAL_POOL_CANCELED);
  require(!stats(pool).occupied_outputs && !stats(pool).pending_sources &&
              stats(pool).host_allocations == expectedAllocations &&
              stats(pool).host_imports == 3,
          "final pooled resources fully drained");
  if (issued) {
    require(mgpuExternalPoolRelease(pool) == MGPU_CONTEXT_LEASE_WOULD_BLOCK,
            "drained renderer/encoder cannot release outstanding producer export");
    child.stop(); // Waits for actual kernel exit of the original owned producer.
    require(mgpuExternalPoolReleaseProducerAfterPeerRelease(
                pool, provided.export_lease) == 0 &&
                mgpuExternalPoolReleaseProducerAfterPeerRelease(
                    pool, provided.export_lease) == MGPU_CONTEXT_LEASE_INVALID,
            "actual peer exit completes only its exact outbound lease once");
  }
  require(mgpuExternalPoolRelease(pool) == 0,
          "release drains owner off GPU/coordinator threads");
  until(
      [&] {
        return mgpuExternalPoolGetLimits(&limits, sizeof(limits)) == 0 &&
               limits.live_pools == baseline.live_pools &&
               limits.reserved_bytes == baseline.reserved_bytes;
      },
      "global quota released only after owner cleanup");
  std::cout << (issued ? "PASS: separate-process host-issued D3D11 pool, initial zero contents, export retention, " :
                        "PASS: separate-process registered D3D11 pool, readiness, ")
            << "pixels, duplicate-handle identity, reuse, consumer fences, "
               "backpressure and retirement\n";
  return 0;
}
#else
int main() {
  std::cout << "Windows integration gate only\n";
  return 0;
}
#endif

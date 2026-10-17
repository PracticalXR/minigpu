#include "../include/minigpu.h"
#include "../include/minigpu_context_owner.h"
#include "../include/dart_port.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

extern "C" mgpu::MGPU minigpu;

namespace {

struct VertexAttributeInfo {
  uint32_t location, offset, size;
  WGPUVertexFormat format;
};
struct VertexLayoutInfo {
  uint32_t stride, step;
  std::vector<VertexAttributeInfo> attributes;
};
struct BindingInfo {
  uint32_t slot, kind, visibility, dimension, minSize;
};

struct BindingKeyEntry {
  uint32_t binding = 0;
  WGPUBuffer buffer = nullptr;
  uint64_t offset = 0, size = 0;
  WGPUTextureView textureView = nullptr;
  WGPUSampler sampler = nullptr;

  bool operator==(const BindingKeyEntry &other) const {
    return binding == other.binding && buffer == other.buffer &&
           offset == other.offset && size == other.size &&
           textureView == other.textureView && sampler == other.sampler;
  }
};

struct CachedBindGroup {
  std::vector<BindingKeyEntry> key;
  WGPUBindGroup group = nullptr;
  uint64_t lastUse = 0;

  ~CachedBindGroup() {
    // Emscripten's C-handle table may recycle an object id after its wrapper is
    // released even though the browser bind group still retains the JS object.
    // Explicit C-API references keep every raw key handle unique for as long as
    // this cache entry can match it. They are also harmless on native Dawn.
    for (const auto &entry : key) {
      if (entry.buffer)
        wgpuBufferRelease(entry.buffer);
      if (entry.textureView)
        wgpuTextureViewRelease(entry.textureView);
      if (entry.sampler)
        wgpuSamplerRelease(entry.sampler);
    }
  }
};

struct PipelineState {
  mgpu::MGPU *owner = nullptr;
  WGPUDevice device = nullptr;
  WGPURenderPipeline pipeline = nullptr;
  WGPUBindGroupLayout layout = nullptr;
  WGPUTextureFormat colorFormat = WGPUTextureFormat_Undefined;
  bool depthEnabled = false;
  std::vector<VertexLayoutInfo> vertexLayouts;
  std::vector<BindingInfo> bindings;
  std::atomic<int> status{0};
  std::atomic<uint64_t> bindGroupsCreated{0}, bindGroupsReused{0},
      bindGroupsRetired{0};
  uint64_t bindGroupClock = 0;
  std::vector<std::unique_ptr<CachedBindGroup>> bindGroupCache;
  std::mutex mutex;
  std::string diagnostic;
  std::string compileError; // first WGSL error with its position
  // Told once when status leaves 0; taken under [mutex], run outside it.
  std::vector<std::function<void(bool)>> settleWaiters;
#ifdef __EMSCRIPTEN__
  int handleId = 0; // the MGPURenderPipeline* handed out, for the JS hook
#endif
  ~PipelineState() {
    for (auto &cached : bindGroupCache) {
      if (cached->group) {
        wgpuBindGroupRelease(cached->group);
        bindGroupsRetired.fetch_add(1);
      }
    }
    if (pipeline)
      wgpuRenderPipelineRelease(pipeline);
    if (layout)
      wgpuBindGroupLayoutRelease(layout);
    if (device)
      wgpuDeviceRelease(device);
  }
};

using SettleWaiters = std::vector<std::function<void(bool)>>;

// Moves [state] from status 0 to [status] and hands over its waiters; false
// when it had already settled. Under state.mutex.
bool settleLocked(PipelineState &state, int status, SettleWaiters &waiters) {
  if (state.status.load() != 0)
    return false;
  state.status.store(status);
  waiters.swap(state.settleWaiters);
  return true;
}

// Tells the waiters of a pipeline that just settled. Never under a lock: a
// waiter may post to Dart or call straight back in.
void announceSettled(const PipelineState &state, SettleWaiters waiters,
                     bool ok) {
  for (auto &waiter : waiters) {
    try {
      waiter(ok);
    } catch (...) {
    }
  }
#ifdef __EMSCRIPTEN__
  EM_ASM({ var f = Module['mgpuRenderPipelineSettled']; if (f) f($0); },
         state.handleId);
#else
  (void)state;
#endif
}

// Fails a pipeline still compiling, with [message] as its diagnostic.
void settleFailed(PipelineState &state, const char *message) {
  SettleWaiters waiters;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.status.load() != 0)
      return;
    state.diagnostic = message;
    settleLocked(state, 2, waiters);
  }
  announceSettled(state, std::move(waiters), false);
}

// Closes [state] (status 3); one still compiling settles now, not ok.
void closePipeline(PipelineState &state) {
  SettleWaiters waiters;
  bool settled;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    settled = settleLocked(state, 3, waiters);
    state.status.store(3);
  }
  if (settled)
    announceSettled(state, std::move(waiters), false);
}

// Every pipeline still referenced, so a context teardown can close its own.
struct PipelineRegistry {
  std::mutex mutex;
  std::vector<std::weak_ptr<PipelineState>> pipelines;
  size_t pruneAt = 64;
};

// Leaked on purpose: the process-global context can be torn down after
// static destructors have run.
PipelineRegistry &pipelineRegistry() {
  static auto *registry = new PipelineRegistry;
  return *registry;
}

void registerPipeline(const std::shared_ptr<PipelineState> &state) {
  auto &registry = pipelineRegistry();
  std::lock_guard<std::mutex> lock(registry.mutex);
  auto &all = registry.pipelines;
  if (all.size() >= registry.pruneAt) {
    all.erase(std::remove_if(all.begin(), all.end(),
                             [](const std::weak_ptr<PipelineState> &weak) {
                               return weak.expired();
                             }),
              all.end());
    registry.pruneAt = std::max<size_t>(64, all.size() * 2);
  }
  all.push_back(state);
}

constexpr size_t kMaxBindGroupsPerPipeline = 64;

WGPUBindGroup
retainedBindGroup(PipelineState &state,
                  const std::vector<WGPUBindGroupEntry> &entries) {
  std::vector<BindingKeyEntry> key;
  key.reserve(entries.size());
  for (const auto &entry : entries) {
    key.push_back({entry.binding, entry.buffer, entry.offset, entry.size,
                   entry.textureView, entry.sampler});
  }
  const uint64_t use = ++state.bindGroupClock;
  for (auto &cached : state.bindGroupCache) {
    if (cached->key == key) {
      cached->lastUse = use;
      state.bindGroupsReused.fetch_add(1);
      return cached->group;
    }
  }

  WGPUBindGroupDescriptor descriptor{};
  descriptor.layout = state.layout;
  descriptor.entryCount = entries.size();
  descriptor.entries = entries.data();
  WGPUBindGroup group = wgpuDeviceCreateBindGroup(state.device, &descriptor);
  if (!group)
    return nullptr;

  if (state.bindGroupCache.size() >= kMaxBindGroupsPerPipeline) {
    auto oldest = std::min_element(
        state.bindGroupCache.begin(), state.bindGroupCache.end(),
        [](const auto &a, const auto &b) { return a->lastUse < b->lastUse; });
    wgpuBindGroupRelease((*oldest)->group);
    state.bindGroupsRetired.fetch_add(1);
    state.bindGroupCache.erase(oldest);
  }
  auto cached = std::make_unique<CachedBindGroup>();
  cached->key = std::move(key);
  for (const auto &entry : cached->key) {
    if (entry.buffer)
      wgpuBufferAddRef(entry.buffer);
    if (entry.textureView)
      wgpuTextureViewAddRef(entry.textureView);
    if (entry.sampler)
      wgpuSamplerAddRef(entry.sampler);
  }
  cached->group = group;
  cached->lastUse = use;
  state.bindGroupCache.push_back(std::move(cached));
  state.bindGroupsCreated.fetch_add(1);
  return group;
}

struct MGPURenderPipelineImpl {
  std::shared_ptr<PipelineState> state;
};

struct ParsedPipeline {
  uint32_t colorCode, blend, topology, frontFace, cull;
  bool depthEnabled, depthWrite;
  uint32_t depthCompare;
  std::vector<VertexLayoutInfo> vertices;
  std::vector<BindingInfo> bindings;
};

WGPUTextureFormat colorFormat(uint32_t c) {
  switch (c) {
  case 1:
    return WGPUTextureFormat_RGBA8Unorm;
  case 2:
    return WGPUTextureFormat_BGRA8Unorm;
  case 3:
    return WGPUTextureFormat_RGBA16Float;
  case 6:
    return WGPUTextureFormat_RGBA8UnormSrgb;
  case 7:
    return WGPUTextureFormat_BGRA8UnormSrgb;
  default:
    return WGPUTextureFormat_Undefined;
  }
}

bool identifier(const char *s) {
  if (!s)
    return false;
  size_t n = strnlen(s, 65);
  if (!n || n > 64 ||
      !((s[0] >= 'A' && s[0] <= 'Z') || (s[0] >= 'a' && s[0] <= 'z') ||
        s[0] == '_'))
    return false;
  for (size_t i = 1; i < n; ++i)
    if (!((s[i] >= 'A' && s[i] <= 'Z') || (s[i] >= 'a' && s[i] <= 'z') ||
          (s[i] >= '0' && s[i] <= '9') || s[i] == '_'))
      return false;
  return true;
}

bool validUtf8(const char *text, size_t size) {
  const auto *bytes = reinterpret_cast<const uint8_t *>(text);
  for (size_t i = 0; i < size;) {
    const uint8_t first = bytes[i++];
    if (first < 0x80)
      continue;
    uint32_t value = 0;
    size_t continuation = 0;
    uint32_t minimum = 0;
    if (first >= 0xc2 && first <= 0xdf) {
      value = first & 0x1f;
      continuation = 1;
      minimum = 0x80;
    } else if (first >= 0xe0 && first <= 0xef) {
      value = first & 0x0f;
      continuation = 2;
      minimum = 0x800;
    } else if (first >= 0xf0 && first <= 0xf4) {
      value = first & 0x07;
      continuation = 3;
      minimum = 0x10000;
    } else {
      return false;
    }
    if (continuation > size - i)
      return false;
    for (size_t j = 0; j < continuation; ++j) {
      const uint8_t next = bytes[i++];
      if ((next & 0xc0) != 0x80)
        return false;
      value = (value << 6) | (next & 0x3f);
    }
    if (value < minimum || value > 0x10ffff ||
        (value >= 0xd800 && value <= 0xdfff))
      return false;
  }
  return true;
}

bool vertexFormat(uint32_t code, WGPUVertexFormat &format, uint32_t &size,
                  uint32_t &alignment) {
  alignment = 4;
  switch (code) {
  case 0:
    format = WGPUVertexFormat_Float32;
    size = 4;
    return true;
  case 1:
    format = WGPUVertexFormat_Float32x2;
    size = 8;
    return true;
  case 2:
    format = WGPUVertexFormat_Float32x3;
    size = 12;
    return true;
  case 3:
    format = WGPUVertexFormat_Float32x4;
    size = 16;
    return true;
  case 4:
    format = WGPUVertexFormat_Uint32;
    size = 4;
    return true;
  case 5:
    format = WGPUVertexFormat_Uint32x2;
    size = 8;
    return true;
  case 6:
    format = WGPUVertexFormat_Uint32x4;
    size = 16;
    return true;
  case 7:
    format = WGPUVertexFormat_Sint32;
    size = 4;
    return true;
  case 8:
    format = WGPUVertexFormat_Unorm8x4;
    size = 4;
    return true;
  default:
    return false;
  }
}

bool parsePipeline(const uint32_t *w, int count, WGPUDevice device,
                   ParsedPipeline &out) {
  if (!w || count < 11 || count > 4096 || w[0] != 1)
    return false;
  if (colorFormat(w[1]) == WGPUTextureFormat_Undefined || w[2] > 3 ||
      w[3] > 2 || w[4] > 1 || w[5] > 2 || w[6] > 1 || w[7] > 1 || w[8] < 1 ||
      w[8] > 8 || w[9] > 8 || w[10] > 32 || (!w[6] && w[7]))
    return false;
  out = {w[1], w[2], w[3], w[4], w[5], w[6] != 0, w[7] != 0, w[8], {}, {}};
  size_t p = 11, attrs = 0;
  std::unordered_set<uint32_t> locations, slots;
  WGPULimits limits{};
  if (wgpuDeviceGetLimits(device, &limits) != WGPUStatus_Success ||
      w[9] > limits.maxVertexBuffers)
    return false;
  for (uint32_t i = 0; i < w[9]; ++i) {
    if (p + 3 > static_cast<size_t>(count))
      return false;
    VertexLayoutInfo v{w[p], w[p + 1], {}};
    uint32_t ac = w[p + 2];
    p += 3;
    if (v.stride < 4 || v.stride > 2048 || (v.stride & 3) || v.step > 1 ||
        ac > 16 || attrs + ac > 16 ||
        v.stride > limits.maxVertexBufferArrayStride)
      return false;
    for (uint32_t j = 0; j < ac; ++j) {
      if (p + 3 > static_cast<size_t>(count))
        return false;
      uint32_t size, align;
      WGPUVertexFormat f;
      if (w[p] > 31 || !locations.insert(w[p]).second ||
          !vertexFormat(w[p + 2], f, size, align) || (w[p + 1] % align) ||
          w[p + 1] > v.stride || size > v.stride - w[p + 1])
        return false;
      v.attributes.push_back({w[p], w[p + 1], size, f});
      p += 3;
      ++attrs;
    }
    out.vertices.push_back(std::move(v));
  }
  if (attrs > limits.maxVertexAttributes)
    return false;
  std::array<std::array<uint32_t, 5>, 2> stage{};
  for (uint32_t i = 0; i < w[10]; ++i) {
    if (p + 5 > static_cast<size_t>(count))
      return false;
    BindingInfo b{w[p], w[p + 1], w[p + 2], w[p + 3], w[p + 4]};
    p += 5;
    if (b.slot > 31 || !slots.insert(b.slot).second || b.kind > 7 ||
        b.visibility < 1 || b.visibility > 3)
      return false;
    if (b.kind <= 1) {
      if (!b.minSize || (b.minSize & 3) || b.minSize > 2147483644u ||
          b.dimension != 0)
        return false;
    } else if (b.kind <= 4) {
      if (b.minSize || b.dimension > 3)
        return false;
    } else if (b.minSize || b.dimension != 0)
      return false;
    for (int s = 0; s < 2; ++s)
      if (b.visibility & (1u << s)) {
        if (b.kind <= 1)
          ++stage[s][b.kind == 0 ? 0 : 1];
        else if (b.kind <= 4)
          ++stage[s][2];
        else
          ++stage[s][3];
      }
    out.bindings.push_back(b);
  }
  if (p != static_cast<size_t>(count) ||
      out.bindings.size() > limits.maxBindingsPerBindGroup)
    return false;
  for (auto &s : stage)
    if (s[0] > limits.maxUniformBuffersPerShaderStage ||
        s[1] > limits.maxStorageBuffersPerShaderStage ||
        s[2] > limits.maxSampledTexturesPerShaderStage ||
        s[3] > limits.maxSamplersPerShaderStage)
      return false;
  return true;
}

WGPUTextureViewDimension viewDimension(uint32_t d) {
  static const WGPUTextureViewDimension v[] = {
      WGPUTextureViewDimension_2D, WGPUTextureViewDimension_2DArray,
      WGPUTextureViewDimension_Cube, WGPUTextureViewDimension_CubeArray};
  return v[d];
}

WGPUCompareFunction compare(uint32_t c) {
  static const WGPUCompareFunction v[] = {
      WGPUCompareFunction_Undefined, WGPUCompareFunction_Never,
      WGPUCompareFunction_Less,      WGPUCompareFunction_Equal,
      WGPUCompareFunction_LessEqual, WGPUCompareFunction_Greater,
      WGPUCompareFunction_NotEqual,  WGPUCompareFunction_GreaterEqual,
      WGPUCompareFunction_Always};
  return v[c];
}

void pipelineCallback(WGPUCreatePipelineAsyncStatus result,
                      WGPURenderPipeline pipeline, WGPUStringView message,
                      void *userdata1, void *) {
  std::unique_ptr<std::shared_ptr<PipelineState>> box(
      static_cast<std::shared_ptr<PipelineState> *>(userdata1));
  auto state = *box;
  const bool ok = result == WGPUCreatePipelineAsyncStatus_Success && pipeline;
  SettleWaiters waiters;
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->status.load() != 0) {
      if (pipeline)
        wgpuRenderPipelineRelease(pipeline);
      return;
    }
    if (ok) {
      state->pipeline = pipeline;
    } else {
      if (pipeline)
        wgpuRenderPipelineRelease(pipeline);
      if (!state->compileError.empty())
        state->diagnostic = state->compileError;
      else if (message.data && message.length)
        // WGPU_STRLEN marks a NUL-terminated message, which may be shorter.
        state->diagnostic.assign(message.data,
                                 message.length == WGPU_STRLEN
                                     ? strnlen(message.data, 512)
                                     : std::min<size_t>(message.length, 512));
    }
    settleLocked(*state, ok ? 1 : 2, waiters);
  }
  announceSettled(*state, std::move(waiters), ok);
}

// Keeps Tint's first error with its line and column: an invalid module
// otherwise surfaces only as "[Invalid ShaderModule] is invalid".
void compilationInfoCallback(WGPUCompilationInfoRequestStatus result,
                             const WGPUCompilationInfo *info, void *userdata1,
                             void *) {
  std::unique_ptr<std::shared_ptr<PipelineState>> box(
      static_cast<std::shared_ptr<PipelineState> *>(userdata1));
  if (result != WGPUCompilationInfoRequestStatus_Success || !info)
    return;
  for (size_t i = 0; i < info->messageCount; ++i) {
    const WGPUCompilationMessage &m = info->messages[i];
    if (m.type != WGPUCompilationMessageType_Error)
      continue;
    std::string text = "line " + std::to_string(m.lineNum) + ":" +
                       std::to_string(m.linePos) + ": ";
    if (m.message.data)
      text.append(m.message.data, m.message.length == WGPU_STRLEN
                                      ? strlen(m.message.data)
                                      : m.message.length);
    auto &state = **box;
    std::lock_guard<std::mutex> lock(state.mutex);
    state.compileError = text.substr(0, 512);
    // On the web this can arrive after the pipeline already failed.
    if (state.status.load() == 2)
      state.diagnostic = state.compileError;
    return;
  }
}

void beginCompile(const std::shared_ptr<PipelineState> &state,
                  ParsedPipeline parsed, std::string code, std::string vs,
                  std::string fs) {
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->status.load() == 3)
      return;
  }
  auto fail = [&](const char *message) { settleFailed(*state, message); };
  WGPUShaderSourceWGSL source{};
  source.chain.sType = WGPUSType_ShaderSourceWGSL;
  source.code = {code.data(), code.size()};
  WGPUShaderModuleDescriptor smd{};
  smd.nextInChain = &source.chain;
  WGPUShaderModule module = wgpuDeviceCreateShaderModule(state->device, &smd);
  if (!module) {
    fail("shader module creation failed");
    return;
  }
  // Dawn reports the (already known) messages inline, so a WGSL error fails
  // here with its position instead of reaching pipeline creation.
  WGPUCompilationInfoCallbackInfo ci{};
  ci.mode = WGPUCallbackMode_AllowSpontaneous;
  ci.callback = compilationInfoCallback;
  ci.userdata1 = new std::shared_ptr<PipelineState>(state);
  wgpuShaderModuleGetCompilationInfo(module, ci);
  std::string compileError;
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    compileError = state->compileError;
  }
  if (!compileError.empty()) {
    wgpuShaderModuleRelease(module);
    fail(compileError.c_str());
    return;
  }
  std::vector<WGPUBindGroupLayoutEntry> les;
  for (auto &b : parsed.bindings) {
    WGPUBindGroupLayoutEntry e{};
    e.binding = b.slot;
    e.visibility =
        (b.visibility & 1 ? WGPUShaderStage_Vertex : WGPUShaderStage_None) |
        (b.visibility & 2 ? WGPUShaderStage_Fragment : WGPUShaderStage_None);
    if (b.kind <= 1) {
      e.buffer.type = b.kind ? WGPUBufferBindingType_ReadOnlyStorage
                             : WGPUBufferBindingType_Uniform;
      e.buffer.minBindingSize = b.minSize;
    } else if (b.kind <= 4) {
      e.texture.sampleType = b.kind == 2 ? WGPUTextureSampleType_Float
                             : b.kind == 3
                                 ? WGPUTextureSampleType_UnfilterableFloat
                                 : WGPUTextureSampleType_Depth;
      e.texture.viewDimension = viewDimension(b.dimension);
    } else
      e.sampler.type = b.kind == 5   ? WGPUSamplerBindingType_Filtering
                       : b.kind == 6 ? WGPUSamplerBindingType_NonFiltering
                                     : WGPUSamplerBindingType_Comparison;
    les.push_back(e);
  }
  WGPUBindGroupLayoutDescriptor ld{};
  ld.entryCount = les.size();
  ld.entries = les.data();
  WGPUBindGroupLayout layout =
      wgpuDeviceCreateBindGroupLayout(state->device, &ld);
  if (!layout) {
    wgpuShaderModuleRelease(module);
    fail("bind group layout creation failed");
    return;
  }
  WGPUPipelineLayoutDescriptor pld{};
  pld.bindGroupLayoutCount = 1;
  pld.bindGroupLayouts = &layout;
  WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(state->device, &pld);
  if (!pl) {
    wgpuBindGroupLayoutRelease(layout);
    wgpuShaderModuleRelease(module);
    fail("pipeline layout creation failed");
    return;
  }
  std::vector<std::vector<WGPUVertexAttribute>> attrs(parsed.vertices.size());
  std::vector<WGPUVertexBufferLayout> vbl(parsed.vertices.size());
  for (size_t i = 0; i < parsed.vertices.size(); ++i) {
    for (auto &a : parsed.vertices[i].attributes) {
      WGPUVertexAttribute attribute{};
      attribute.format = a.format;
      attribute.offset = a.offset;
      attribute.shaderLocation = a.location;
      attrs[i].push_back(attribute);
    }
    vbl[i].stepMode = parsed.vertices[i].step ? WGPUVertexStepMode_Instance
                                              : WGPUVertexStepMode_Vertex;
    vbl[i].arrayStride = parsed.vertices[i].stride;
    vbl[i].attributeCount = attrs[i].size();
    vbl[i].attributes = attrs[i].data();
  }
  WGPUBlendState blend{};
  blend.color.operation = WGPUBlendOperation_Add;
  blend.alpha.operation = WGPUBlendOperation_Add;
  if (parsed.blend == 1) {
    blend.color.srcFactor = WGPUBlendFactor_SrcAlpha;
    blend.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
    blend.alpha.srcFactor = WGPUBlendFactor_One;
    blend.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
  } else {
    blend.color.srcFactor = WGPUBlendFactor_One;
    blend.color.dstFactor = parsed.blend == 3
                                ? WGPUBlendFactor_One
                                : WGPUBlendFactor_OneMinusSrcAlpha;
    blend.alpha = blend.color;
  }
  WGPUColorTargetState target{};
  target.format = colorFormat(parsed.colorCode);
  target.writeMask = WGPUColorWriteMask_All;
  if (parsed.blend)
    target.blend = &blend;
  WGPUFragmentState frag{};
  frag.module = module;
  frag.entryPoint = {fs.data(), fs.size()};
  frag.targetCount = 1;
  frag.targets = &target;
  WGPUDepthStencilState depth{};
  depth.format = WGPUTextureFormat_Depth32Float;
  depth.depthWriteEnabled =
      parsed.depthWrite ? WGPUOptionalBool_True : WGPUOptionalBool_False;
  depth.depthCompare = compare(parsed.depthCompare);
  depth.stencilReadMask = 0xffffffffu;
  depth.stencilWriteMask = 0xffffffffu;
  WGPURenderPipelineDescriptor pd{};
  pd.layout = pl;
  pd.vertex.module = module;
  pd.vertex.entryPoint = {vs.data(), vs.size()};
  pd.vertex.bufferCount = vbl.size();
  pd.vertex.buffers = vbl.data();
  pd.fragment = &frag;
  pd.primitive.topology =
      parsed.topology == 0   ? WGPUPrimitiveTopology_TriangleList
      : parsed.topology == 1 ? WGPUPrimitiveTopology_LineList
                             : WGPUPrimitiveTopology_PointList;
  pd.primitive.frontFace =
      parsed.frontFace ? WGPUFrontFace_CW : WGPUFrontFace_CCW;
  pd.primitive.cullMode = parsed.cull == 0   ? WGPUCullMode_None
                          : parsed.cull == 1 ? WGPUCullMode_Front
                                             : WGPUCullMode_Back;
  pd.multisample.count = 1;
  pd.multisample.mask = 0xffffffffu;
  pd.multisample.alphaToCoverageEnabled = false;
  if (parsed.depthEnabled)
    pd.depthStencil = &depth;
  auto *box = new std::shared_ptr<PipelineState>(state);
  WGPUCreateRenderPipelineAsyncCallbackInfo cb{};
  cb.mode = WGPUCallbackMode_AllowSpontaneous;
  cb.callback = pipelineCallback;
  cb.userdata1 = box;
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    state->layout = layout;
  }
  wgpuDeviceCreateRenderPipelineAsync(state->device, &pd, cb);
  wgpuPipelineLayoutRelease(pl);
  wgpuShaderModuleRelease(module);
}

MGPURenderPipeline *createPipeline(mgpu::MGPU &owner, const char *wgsl,
                                   const char *vs, const char *fs,
                                   const uint32_t *desc, int words) {
  if (!wgsl || !identifier(vs) || !identifier(fs))
    return nullptr;
  size_t n = strnlen(wgsl, 1048577);
  if (!n || n > 1048576 || !validUtf8(wgsl, n))
    return nullptr;
  WGPUDevice device = owner.retainReadyDevice();
  if (!device)
    return nullptr;
  ParsedPipeline parsed{};
  if (!parsePipeline(desc, words, device, parsed)) {
    wgpuDeviceRelease(device);
    return nullptr;
  }
  try {
    auto state = std::make_shared<PipelineState>();
    state->owner = &owner;
    state->device = device;
    device = nullptr;
    state->colorFormat = colorFormat(parsed.colorCode);
    state->depthEnabled = parsed.depthEnabled;
    state->vertexLayouts = parsed.vertices;
    state->bindings = parsed.bindings;
    auto handle =
        std::make_unique<MGPURenderPipelineImpl>(MGPURenderPipelineImpl{state});
#ifdef __EMSCRIPTEN__
    // Before the compile task: the web runs it inline, so it can settle here.
    state->handleId = static_cast<int>(reinterpret_cast<intptr_t>(handle.get()));
#endif
    registerPipeline(state);
    std::string code(wgsl, n), ve(vs), fe(fs);
    owner.getWebGPUThread().enqueueAsync(
        [state, parsed = std::move(parsed), code = std::move(code),
         ve = std::move(ve), fe = std::move(fe)]() mutable {
          try {
            beginCompile(state, std::move(parsed), std::move(code),
                         std::move(ve), std::move(fe));
          } catch (...) {
            settleFailed(*state, "render pipeline preparation failed");
          }
        });
    return reinterpret_cast<MGPURenderPipeline *>(handle.release());
  } catch (...) {
    if (device)
      wgpuDeviceRelease(device);
    return nullptr;
  }
}

struct Command {
  uint32_t op;
  std::array<uint32_t, 6> a{};
};
struct BufferRef {
  mgpu::Buffer *wrapper = nullptr;
  WGPUBuffer buffer = nullptr;
  WGPUBufferUsage usage = WGPUBufferUsage_None;
  uint64_t size = 0;
  BufferRef(mgpu::Buffer *w, WGPUBuffer b, WGPUBufferUsage u, uint64_t s)
      : wrapper(w), buffer(b), usage(u), size(s) {}
  BufferRef(const BufferRef &) = delete;
  BufferRef &operator=(const BufferRef &) = delete;
  BufferRef(BufferRef &&o) noexcept
      : wrapper(o.wrapper), buffer(o.buffer), usage(o.usage), size(o.size) {
    o.buffer = nullptr;
  }
  BufferRef &operator=(BufferRef &&o) noexcept {
    if (this != &o) {
      if (buffer)
        wgpuBufferRelease(buffer);
      wrapper = o.wrapper;
      buffer = o.buffer;
      usage = o.usage;
      size = o.size;
      o.buffer = nullptr;
    }
    return *this;
  }
  ~BufferRef() {
    if (buffer)
      wgpuBufferRelease(buffer);
  }
};

bool range(uint64_t offset, uint64_t size, uint64_t total) {
  return size && offset <= total && size <= total - offset;
}

// Runs [done] once [pipeline] has left status 0 (at once when it already
// has), with true when it became ready. Never under a lock. Throws only
// before [done] was taken, so the caller still owns it then.
void whenSettled(MGPURenderPipeline *pipeline, std::function<void(bool)> done) {
  std::shared_ptr<PipelineState> state;
  if (pipeline)
    state = reinterpret_cast<MGPURenderPipelineImpl *>(pipeline)->state;
  if (state) {
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->status.load() == 0) {
      state->settleWaiters.push_back(std::move(done));
      return;
    }
  }
  done(state && state->status.load() == 1);
}

} // namespace

namespace mgpu {
// From externalOnContextDestroyed: closes (status 3) every pipeline of the
// context just torn down, so status and submit stop at the status check
// instead of reaching that context, and one still compiling settles now.
void renderOnContextDestroyed(const void *context) {
  try {
    std::vector<std::shared_ptr<PipelineState>> closing;
    {
      auto &registry = pipelineRegistry();
      std::lock_guard<std::mutex> lock(registry.mutex);
      auto &all = registry.pipelines;
      closing.reserve(all.size());
      all.erase(std::remove_if(
                    all.begin(), all.end(),
                    [&](const std::weak_ptr<PipelineState> &weak) {
                      auto state = weak.lock();
                      if (state &&
                          static_cast<const void *>(state->owner) != context)
                        return false;
                      if (state)
                        closing.push_back(std::move(state));
                      return true;
                    }),
                all.end());
    }
    for (auto &state : closing)
      closePipeline(*state);
  } catch (...) {
    // Bookkeeping must not fail the teardown that called it.
  }
}
} // namespace mgpu

extern "C" {

uint32_t mgpuRasterResourceVersion(void) { return 1; }
MGPURenderPipeline *mgpuCreateRenderPipeline(const char *w, const char *v,
                                             const char *f, const uint32_t *d,
                                             int n) {
  try {
    return createPipeline(minigpu, w, v, f, d, n);
  } catch (...) {
    return nullptr;
  }
}
MGPURenderPipeline *mgpuContextCreateRenderPipeline(MGPUContextHandle *c,
                                                    const char *w,
                                                    const char *v,
                                                    const char *f,
                                                    const uint32_t *d, int n) {
  try {
    // A destroyed or unknown handle is refused, not dereferenced.
    auto owner = mgpu::retainHandleOwner(c);
    return owner ? createPipeline(owner->gpu(), w, v, f, d, n) : nullptr;
  } catch (...) {
    return nullptr;
  }
}

int mgpuRenderPipelineStatus(MGPURenderPipeline *p) {
  try {
    if (!p)
      return 3;
    auto s = reinterpret_cast<MGPURenderPipelineImpl *>(p)->state;
    if (!s || s->status.load() == 3)
      return 3;
    WGPUDevice d = s->owner->retainReadyDevice();
    bool ok = d && d == s->device;
    if (d)
      wgpuDeviceRelease(d);
    return ok ? s->status.load() : 3;
  } catch (...) {
    return 3;
  }
}
void mgpuRenderPipelineNotifySettledToPort(MGPURenderPipeline *p, int64_t port,
                                           int64_t token) {
  auto post = [port, token](bool ok) { mgpu::completionPost(port, token, ok); };
  try {
    whenSettled(p, post);
  } catch (...) {
    post(false); // thrown before the waiter was taken
  }
}
int mgpuRenderPipelineDiagnostic(MGPURenderPipeline *p, char *out, int cap) {
  try {
    if (!p || cap < 0)
      return 0;
    auto s = reinterpret_cast<MGPURenderPipelineImpl *>(p)->state;
    std::lock_guard<std::mutex> l(s->mutex);
    int n =
        std::min<int>({cap > 0 ? cap - 1 : 0, 512, (int)s->diagnostic.size()});
    if (out && cap > 0) {
      if (n)
        memcpy(out, s->diagnostic.data(), n);
      out[n] = 0;
    }
    return n;
  } catch (...) {
    if (out && cap > 0)
      out[0] = 0;
    return 0;
  }
}
double mgpuRenderPipelineBindingStat(MGPURenderPipeline *p, int selector) {
  try {
    if (!p)
      return 0;
    auto state = reinterpret_cast<MGPURenderPipelineImpl *>(p)->state;
    if (!state)
      return 0;
    const uint64_t created = state->bindGroupsCreated.load();
    const uint64_t retired = state->bindGroupsRetired.load();
    switch (selector) {
    case 0:
      return static_cast<double>(created);
    case 1:
      return static_cast<double>(state->bindGroupsReused.load());
    case 2:
      return static_cast<double>(retired);
    case 3:
      return static_cast<double>(created >= retired ? created - retired : 0);
    default:
      return 0;
    }
  } catch (...) {
    return 0;
  }
}
void mgpuDestroyRenderPipeline(MGPURenderPipeline *p) {
  if (!p)
    return;
  try {
    auto *h = reinterpret_cast<MGPURenderPipelineImpl *>(p);
    if (h->state)
      closePipeline(*h->state); // its waiters run while [p] is still valid
    delete h;
  } catch (...) {
  }
}

int mgpuSubmitRender(MGPURenderPipeline *p, MGPUTextureView *color,
                     MGPUTextureView *depth, const float *clear, int loadFlags,
                     MGPUBuffer *const *buffers, int bufferCount,
                     MGPUTextureView *const *views, int viewCount,
                     MGPUSampler *const *samplers, int samplerCount,
                     const uint32_t *words, int wordCount) {
  try {
    if (!p || !color || !clear || loadFlags < 0 || (loadFlags & ~3) ||
        (!depth && (loadFlags & 2)) || bufferCount < 0 || bufferCount > 256 ||
        viewCount < 0 || viewCount > 256 || samplerCount < 0 ||
        samplerCount > 256 || wordCount < 0 || wordCount > 65536 ||
        ((bufferCount && !buffers) || (viewCount && !views) ||
         (samplerCount && !samplers) || (wordCount && !words)))
      return 1;
    auto state = reinterpret_cast<MGPURenderPipelineImpl *>(p)->state;
    if (!state || state->status.load() != 1)
      return state && state->status.load() == 0 ? 2 : 2;
    WGPUDevice current = state->owner->retainReadyDevice();
    if (!current || current != state->device) {
      if (current)
        wgpuDeviceRelease(current);
      return 2;
    }
    wgpuDeviceRelease(current);
    auto cv = mgpu::retainOwnedTextureViewState(color),
         dv = mgpu::retainOwnedTextureViewState(depth);
    if (!cv)
      return 1;
    if (!cv->isAlive())
      return 2;
    if (cv->textureState->owner != state->owner ||
        cv->textureState->creationDevice != state->device ||
        !(cv->textureState->usage & WGPUTextureUsage_RenderAttachment) ||
        cv->dimension != WGPUTextureViewDimension_2D || cv->mipCount != 1 ||
        cv->layerCount != 1 || cv->textureState->format != state->colorFormat)
      return 1;
    if (state->depthEnabled) {
      if (!dv)
        return 1;
      if (!dv->isAlive())
        return 2;
      if (dv->textureState->owner != state->owner ||
          dv->textureState->creationDevice != state->device ||
          dv->textureState->format != WGPUTextureFormat_Depth32Float ||
          !(dv->textureState->usage & WGPUTextureUsage_RenderAttachment) ||
          dv->dimension != WGPUTextureViewDimension_2D || dv->mipCount != 1 ||
          dv->layerCount != 1 || dv->mipWidth != cv->mipWidth ||
          dv->mipHeight != cv->mipHeight)
        return 1;
    } else if (depth)
      return 1;
    for (int i = 0; i < 5; ++i)
      if (!std::isfinite(clear[i]))
        return 1;
    if (clear[4] < 0 || clear[4] > 1)
      return 1;
    std::vector<BufferRef> br;
    for (int i = 0; i < bufferCount; ++i) {
      if (!buffers[i])
        return 1;
      auto *b = reinterpret_cast<mgpu::Buffer *>(buffers[i]);
      if (&b->getMGPU() != state->owner || !b->bufferData.buffer ||
          b->bufferData.creationDevice != state->device)
        return 1;
    }
    br.reserve(bufferCount);
    for (int i = 0; i < bufferCount; ++i) {
      auto *b = reinterpret_cast<mgpu::Buffer *>(buffers[i]);
      wgpuBufferAddRef(b->bufferData.buffer);
      br.emplace_back(b, b->bufferData.buffer, b->bufferData.usage,
                      b->bufferData.size);
    }
    std::vector<std::shared_ptr<mgpu::OwnedTextureViewState>> vr;
    std::vector<std::shared_ptr<mgpu::OwnedSamplerState>> sr;
    for (int i = 0; i < viewCount; ++i) {
      auto v = mgpu::retainOwnedTextureViewState(views[i]);
      if (!v)
        goto invalid;
      if (!v->isAlive())
        goto stale;
      if (v->textureState->owner != state->owner ||
          v->textureState->creationDevice != state->device ||
          v->textureState == cv->textureState ||
          (dv && v->textureState == dv->textureState))
        goto invalid;
      vr.push_back(v);
    }
    for (int i = 0; i < samplerCount; ++i) {
      auto s = mgpu::retainOwnedSamplerState(samplers[i]);
      if (!s)
        goto invalid;
      if (!s->alive.load())
        goto stale;
      if (s->owner != state->owner || s->creationDevice != state->device)
        goto invalid;
      sr.push_back(s);
    }
    {
      std::vector<Command> cmds;
      size_t x = 0;
      int draws = 0;
      std::vector<bool> vb(state->vertexLayouts.size()),
          bound(state->bindings.size());
      std::vector<uint64_t> vbSize(state->vertexLayouts.size());
      bool index = false;
      uint64_t indexSize = 0;
      uint32_t indexBytes = 0;
      WGPULimits limits{};
      if (wgpuDeviceGetLimits(state->device, &limits) != WGPUStatus_Success)
        goto invalid;
      while (x < (size_t)wordCount) {
        uint32_t op = words[x],
                 len = op == 8   ? 7
                       : op == 7 ? 6
                       : op == 9 ? 5
                                 : (op >= 1 && op <= 6 ? (op == 4 || op == 5 ? 3
                                                          : op == 6          ? 5
                                                                    : 5)
                                                       : 0);
        if (!len || x + len > (size_t)wordCount || cmds.size() >= 4096)
          goto invalid;
        Command c{};
        c.op = op;
        for (uint32_t j = 1; j < len; ++j)
          c.a[j - 1] = words[x + j];
        x += len;
        if (op == 1) {
          if (c.a[0] >= vb.size() || c.a[1] >= br.size() ||
              !(br[c.a[1]].usage & WGPUBufferUsage_Vertex) ||
              !range(c.a[2], c.a[3], br[c.a[1]].size) || c.a[2] & 3)
            goto invalid;
          vb[c.a[0]] = true;
          vbSize[c.a[0]] = c.a[3];
        } else if (op == 2) {
          if (c.a[0] >= br.size() || c.a[3] > 1 ||
              !(br[c.a[0]].usage & WGPUBufferUsage_Index) ||
              !range(c.a[1], c.a[2], br[c.a[0]].size) ||
              c.a[1] % (c.a[3] ? 4 : 2) || c.a[2] % (c.a[3] ? 4 : 2))
            goto invalid;
          index = true;
          indexSize = c.a[2];
          indexBytes = c.a[3] ? 4 : 2;
        } else if (op == 3) {
          auto it = std::find_if(state->bindings.begin(), state->bindings.end(),
                                 [&](auto &b) { return b.slot == c.a[0]; });
          if (it == state->bindings.end() || it->kind > 1 ||
              c.a[1] >= br.size() || !range(c.a[2], c.a[3], br[c.a[1]].size) ||
              (c.a[3] & 3) || c.a[3] < it->minSize ||
              !(br[c.a[1]].usage & (it->kind ? WGPUBufferUsage_Storage
                                             : WGPUBufferUsage_Uniform)) ||
              (c.a[2] % (it->kind ? limits.minStorageBufferOffsetAlignment
                                  : limits.minUniformBufferOffsetAlignment)) ||
              (it->kind ? c.a[3] > limits.maxStorageBufferBindingSize
                        : c.a[3] > limits.maxUniformBufferBindingSize))
            goto invalid;
          bound[it - state->bindings.begin()] = true;
        } else if (op == 4) {
          auto it = std::find_if(state->bindings.begin(), state->bindings.end(),
                                 [&](auto &b) { return b.slot == c.a[0]; });
          if (it == state->bindings.end() || it->kind < 2 || it->kind > 4 ||
              c.a[1] >= vr.size())
            goto invalid;
          auto &v = vr[c.a[1]];
          if (v->dimension != viewDimension(it->dimension) ||
              !(v->textureState->usage & WGPUTextureUsage_TextureBinding) ||
              (it->kind == 4) != v->textureState->depth ||
              (it->kind == 2 && !v->textureState->filterable))
            goto invalid;
          bound[it - state->bindings.begin()] = true;
        } else if (op == 5) {
          auto it = std::find_if(state->bindings.begin(), state->bindings.end(),
                                 [&](auto &b) { return b.slot == c.a[0]; });
          if (it == state->bindings.end() || it->kind < 5 ||
              c.a[1] >= sr.size() ||
              (it->kind == 7) != sr[c.a[1]]->comparison ||
              (it->kind == 6 && !sr[c.a[1]]->allNearest))
            goto invalid;
          bound[it - state->bindings.begin()] = true;
        } else if (op == 6 || op == 7) {
          const uint64_t firstElement = c.a[2];
          const uint64_t firstInstance = c.a[op == 6 ? 3 : 4];
          if (++draws > 1024 || !c.a[0] || !c.a[1] || c.a[0] > 16777216 ||
              c.a[1] > 1048576 ||
              firstElement + c.a[0] > uint64_t(UINT32_MAX) + 1 ||
              firstInstance + c.a[1] > uint64_t(UINT32_MAX) + 1 ||
              std::find(vb.begin(), vb.end(), false) != vb.end() ||
              std::find(bound.begin(), bound.end(), false) != bound.end() ||
              (op == 7 &&
               (!index || uint64_t(c.a[2]) + c.a[0] > indexSize / indexBytes)))
            goto invalid;
          for (size_t vi = 0; vi < state->vertexLayouts.size(); ++vi) {
            auto &vl = state->vertexLayouts[vi];
            uint64_t maxEnd = 0;
            for (auto &a : vl.attributes)
              maxEnd = std::max(maxEnd, uint64_t(a.offset) + a.size);
            uint64_t first = vl.step ? (uint64_t)c.a[op == 6 ? 3 : 4]
                                     : (op == 6 ? (uint64_t)c.a[2] : 0);
            uint64_t count = vl.step ? c.a[1] : (op == 6 ? c.a[0] : 1);
            if (count &&
                ((first > UINT64_MAX - (count - 1)) ||
                 first + count - 1 > (UINT64_MAX - maxEnd) / vl.stride ||
                 ((first + count - 1) * vl.stride + maxEnd) > vbSize[vi]))
              goto invalid;
          }
        } else if (op == 8) {
          float f[6];
          memcpy(f, c.a.data(), 24);
          if (!std::all_of(f, f + 6,
                           [](float z) { return std::isfinite(z); }) ||
              f[0] < 0 || f[1] < 0 || f[2] <= 0 || f[3] <= 0 ||
              f[0] + f[2] > cv->mipWidth || f[1] + f[3] > cv->mipHeight ||
              f[4] < 0 || f[5] > 1 || f[4] > f[5])
            goto invalid;
        } else if (op == 9) {
          if (!c.a[2] || !c.a[3] || c.a[0] > cv->mipWidth ||
              c.a[1] > cv->mipHeight || c.a[2] > cv->mipWidth - c.a[0] ||
              c.a[3] > cv->mipHeight - c.a[1])
            goto invalid;
        }
        cmds.push_back(c);
      }
      std::array<float, 5> clears;
      std::copy(clear, clear + 5, clears.begin());
      auto bufferPins = std::make_shared<std::vector<BufferRef>>(std::move(br));
      state->owner->getWebGPUThread().enqueueAsync(
          [state, cv, dv, bufferPins, vr = std::move(vr), sr = std::move(sr),
           cmds = std::move(cmds), clears, loadFlags]() mutable {
            auto &br = *bufferPins;
            // Only a device change drops an admitted pass. The task pins the
            // pipeline and every resource, and their destroys queue behind
            // it, so wrappers destroyed after submit are still encoded.
            WGPUDevice d = state->owner->retainReadyDevice();
            if (!d || d != state->device) {
              if (d)
                wgpuDeviceRelease(d);
              return;
            }
            wgpuDeviceRelease(d);
            mgpu::lock_guard<mgpu::mutex> lock(state->owner->getGpuMutex());
            state->owner->flushBatchLocked();
            WGPUCommandEncoder enc =
                wgpuDeviceCreateCommandEncoder(state->device, nullptr);
            if (!enc)
              return;
            WGPURenderPassColorAttachment ca{};
            ca.view = cv->view;
            ca.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
            ca.loadOp = (loadFlags & 1) ? WGPULoadOp_Load : WGPULoadOp_Clear;
            ca.storeOp = WGPUStoreOp_Store;
            ca.clearValue = {clears[0], clears[1], clears[2], clears[3]};
            WGPURenderPassDepthStencilAttachment da{};
            if (dv) {
              da.view = dv->view;
              da.depthLoadOp =
                  (loadFlags & 2) ? WGPULoadOp_Load : WGPULoadOp_Clear;
              da.depthStoreOp = WGPUStoreOp_Store;
              da.depthClearValue = clears[4];
              da.depthReadOnly = false;
              da.stencilLoadOp = WGPULoadOp_Undefined;
              da.stencilStoreOp = WGPUStoreOp_Undefined;
              da.stencilReadOnly = true;
            }
            WGPURenderPassDescriptor rd{};
            rd.colorAttachmentCount = 1;
            rd.colorAttachments = &ca;
            rd.depthStencilAttachment = dv ? &da : nullptr;
            auto pass = wgpuCommandEncoderBeginRenderPass(enc, &rd);
            if (!pass) {
              wgpuCommandEncoderRelease(enc);
              return;
            }
            wgpuRenderPassEncoderSetPipeline(pass, state->pipeline);
            wgpuRenderPassEncoderSetViewport(pass, 0, 0, (float)cv->mipWidth,
                                             (float)cv->mipHeight, 0, 1);
            wgpuRenderPassEncoderSetScissorRect(pass, 0, 0, cv->mipWidth,
                                                cv->mipHeight);
            std::unordered_map<uint32_t, WGPUBindGroupEntry> binds;
            bool dirty = true;
            bool encodeOk = true;
            for (auto &c : cmds) {
              if (c.op == 1)
                wgpuRenderPassEncoderSetVertexBuffer(
                    pass, c.a[0], br[c.a[1]].buffer, c.a[2], c.a[3]);
              else if (c.op == 2)
                wgpuRenderPassEncoderSetIndexBuffer(
                    pass, br[c.a[0]].buffer,
                    c.a[3] ? WGPUIndexFormat_Uint32 : WGPUIndexFormat_Uint16,
                    c.a[1], c.a[2]);
              else if (c.op == 3) {
                WGPUBindGroupEntry e{};
                e.binding = c.a[0];
                e.buffer = br[c.a[1]].buffer;
                e.offset = c.a[2];
                e.size = c.a[3];
                binds[e.binding] = e;
                dirty = true;
              } else if (c.op == 4) {
                WGPUBindGroupEntry e{};
                e.binding = c.a[0];
                e.textureView = vr[c.a[1]]->view;
                binds[e.binding] = e;
                dirty = true;
              } else if (c.op == 5) {
                WGPUBindGroupEntry e{};
                e.binding = c.a[0];
                e.sampler = sr[c.a[1]]->sampler;
                binds[e.binding] = e;
                dirty = true;
              } else if (c.op == 8) {
                float f[6];
                memcpy(f, c.a.data(), 24);
                wgpuRenderPassEncoderSetViewport(pass, f[0], f[1], f[2], f[3],
                                                 f[4], f[5]);
              } else if (c.op == 9)
                wgpuRenderPassEncoderSetScissorRect(pass, c.a[0], c.a[1],
                                                    c.a[2], c.a[3]);
              else {
                if (dirty && !state->bindings.empty()) {
                  std::vector<WGPUBindGroupEntry> es;
                  for (auto &b : state->bindings)
                    es.push_back(binds[b.slot]);
                  auto bg = retainedBindGroup(*state, es);
                  if (!bg) {
                    encodeOk = false;
                    break;
                  }
                  wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
                  dirty = false;
                }
                if (c.op == 6)
                  wgpuRenderPassEncoderDraw(pass, c.a[0], c.a[1], c.a[2],
                                            c.a[3]);
                else
                  wgpuRenderPassEncoderDrawIndexed(pass, c.a[0], c.a[1], c.a[2],
                                                   static_cast<int32_t>(c.a[3]),
                                                   c.a[4]);
              }
            }
            wgpuRenderPassEncoderEnd(pass);
            wgpuRenderPassEncoderRelease(pass);
            if (!encodeOk) {
              wgpuCommandEncoderRelease(enc);
              return;
            }
            auto cb = wgpuCommandEncoderFinish(enc, nullptr);
            wgpuCommandEncoderRelease(enc);
            if (cb) {
              auto q = state->owner->tryGetQueue();
              if (q)
                wgpuQueueSubmit(q, 1, &cb);
              wgpuCommandBufferRelease(cb);
            }
          });
      return 0;
    }
  invalid:
    return 1;
  stale:
    return 2;
  } catch (...) {
    return 3;
  }
}

} // extern C

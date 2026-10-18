#include "../include/minigpu.h"
#include "../include/compute_shader.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

// Pipelines compile on the GPU worker with the GPU mutex released, can be built
// before the first dispatch, and buffer-only bind groups come from a small
// per-shader cache. Every dispatch result is read back, so a stale pipeline or
// a cached bind group that names the wrong buffers shows up as wrong numbers.

namespace {

constexpr int kF32 = 1; // mgpuCreateBuffer data type code
constexpr int kCount = 64;
constexpr int kBytes = kCount * static_cast<int>(sizeof(float));
constexpr int kStorage = 0; // mgpuSetBufferRange kinds
constexpr int kReadOnlyStorage = 1;

const char *kDouble = R"(
@group(0) @binding(0) var<storage, read_write> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i < arrayLength(&inp)) { out[i] = inp[i] * 2.0 + 1.0; }
}
)";

const char *kTriple = R"(
@group(0) @binding(0) var<storage, read_write> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i < arrayLength(&inp)) { out[i] = inp[i] * 3.0 + 1.0; }
}
)";

// Read-only input: only a read-only binding kind matches it.
const char *kDoubleReadOnly = R"(
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i < arrayLength(&inp)) { out[i] = inp[i] * 2.0 + 1.0; }
}
)";

// Every dispatch adds into its output, so the total proves each one ran
// against its own buffers.
const char *kAccumulate = R"(
@group(0) @binding(0) var<storage, read_write> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i < arrayLength(&inp)) { out[i] = out[i] + inp[i]; }
}
)";

const char *kNotWgsl = "fn main( { this is not WGSL";

int failures = 0;

void check(bool ok, const char *name) {
  std::cout << (ok ? "PASS " : "FAIL ") << name << std::endl;
  if (!ok) ++failures;
}

void require(bool condition, const char *message) {
  if (!condition) {
    std::cout << "FAIL " << message << std::endl;
    std::exit(1);
  }
}

// data[i] = slope * i + intercept.
void fill(MGPUBuffer *buffer, float slope, float intercept = 0.0f) {
  float data[kCount];
  for (int i = 0; i < kCount; ++i)
    data[i] = slope * static_cast<float>(i) + intercept;
  mgpuWriteFloat(buffer, data, sizeof(data));
}

// Zeroed explicitly: "never written" must not depend on allocation details.
MGPUBuffer *makeBuffer() {
  MGPUBuffer *buffer = mgpuCreateBuffer(kBytes, kF32);
  require(buffer != nullptr, "buffer creation failed");
  fill(buffer, 0.0f);
  return buffer;
}

// out[i] == slope * i + intercept for every i. Small integers, so exact.
bool matches(MGPUBuffer *out, float slope, float intercept) {
  float data[kCount] = {};
  mgpuReadSync(out, data, sizeof(data), 0);
  for (int i = 0; i < kCount; ++i) {
    if (data[i] != slope * static_cast<float>(i) + intercept) return false;
  }
  return true;
}

bool zeros(MGPUBuffer *out) { return matches(out, 0.0f, 0.0f); }

MGPUComputeShader *makeShader(const char *kernel) {
  MGPUComputeShader *shader = mgpuCreateComputeShader();
  require(shader != nullptr, "shader creation failed");
  mgpuLoadKernel(shader, kernel);
  return shader;
}

const mgpu::ComputeShader &native(MGPUComputeShader *shader) {
  return *reinterpret_cast<const mgpu::ComputeShader *>(shader);
}

bool bind(MGPUComputeShader *shader, MGPUBuffer *in, MGPUBuffer *out,
          int inKind = kStorage) {
  return mgpuSetBufferRange(shader, 0, in, inKind, 0, kBytes) == 0 &&
         mgpuSetBufferRange(shader, 1, out, kStorage, 0, kBytes) == 0;
}

// Dispatches are queued; drain so they are recorded before the read flushes.
void run(MGPUComputeShader *shader) {
  mgpuDispatch(shader, 1, 1, 1);
  mgpuDrainWorkQueue();
}

// A long dependent chain, so the backend compile takes measurable time.
std::string largeKernel(int statements) {
  std::string kernel =
      "// pipeline async test: large kernel\n"
      "@group(0) @binding(0) var<storage, read_write> inp: array<f32>;\n"
      "@group(0) @binding(1) var<storage, read_write> out: array<f32>;\n"
      "@compute @workgroup_size(64)\n"
      "fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n"
      "  let i = id.x;\n"
      "  if (i >= arrayLength(&inp)) { return; }\n"
      "  var v = inp[i];\n";
  for (int s = 0; s < statements; ++s) {
    kernel += "  v = fract(sin(v * " + std::to_string(1.0 + s * 0.001) +
              " + " + std::to_string(s) + ".0) * 43758.5453);\n";
  }
  kernel += "  out[i] = v;\n}\n";
  return kernel;
}

double millisSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

} // namespace

int main() {
  mgpuInitializeContext();
  // Fresh compiles on every run, and no test kernel in the user's cache.
  mgpuShaderCacheSetEnabled(0);

  MGPUBuffer *inp = makeBuffer();
  fill(inp, 1.0f);

  // 1. The first dispatch compiles (off the GPU mutex) and writes its output,
  //    on both the batched and the immediate dispatch paths.
  {
    MGPUComputeShader *shader = makeShader(kDouble);
    MGPUBuffer *out = makeBuffer();
    const bool bound = bind(shader, inp, out);
    run(shader);
    check(bound && matches(out, 2.0f, 1.0f),
          "first batched dispatch compiles its pipeline and writes its output");
    fill(out, 0.0f);
    run(shader);
    check(matches(out, 2.0f, 1.0f), "a later dispatch reuses the pipeline");

    MGPUComputeShader *immediate = makeShader(kTriple);
    MGPUBuffer *immediateOut = makeBuffer();
    const bool immediateBound = bind(immediate, inp, immediateOut);
    mgpuDispatchAsync(immediate, 1, 1, 1, nullptr);
    mgpuDrainWorkQueue();
    check(immediateBound && matches(immediateOut, 3.0f, 1.0f),
          "first immediate dispatch compiles its pipeline and submits");
    mgpuDestroyComputeShader(immediate);
    mgpuDestroyComputeShader(shader);
    mgpuDestroyBuffer(immediateOut);
    mgpuDestroyBuffer(out);
  }

  // 2. Build without dispatching, then dispatch.
  {
    MGPUComputeShader *shader = makeShader(kDouble);
    MGPUBuffer *out = makeBuffer();
    const bool bound = bind(shader, inp, out);
    const int built = mgpuComputeShaderBuildPipeline(shader);
    const uint64_t hits = native(shader).bindGroupCacheHits();
    const uint64_t misses = native(shader).bindGroupCacheMisses();
    check(bound && built == 1, "build without dispatch reports the pipeline");
    check(zeros(out), "a build records no dispatch");
    run(shader);
    check(matches(out, 2.0f, 1.0f), "the dispatch after a build writes its output");
    check(native(shader).bindGroupCacheHits() == hits &&
              native(shader).bindGroupCacheMisses() == misses,
          "the dispatch after a build compiled nothing and built no bind group");
    check(mgpuComputeShaderBuildPipeline(shader) == 1,
          "a repeated build answers from the built pipeline");

    // The port form queues like a dispatch. Without a Dart port its post is a
    // no-op, so the dispatch queued behind it is the evidence.
    MGPUComputeShader *ported = makeShader(kTriple);
    MGPUBuffer *portedOut = makeBuffer();
    const bool portedBound = bind(ported, inp, portedOut);
    mgpuComputeShaderBuildPipelineAsyncToPort(ported, 0, 1);
    run(ported);
    check(portedBound && matches(portedOut, 3.0f, 1.0f),
          "the port form builds ahead of the dispatch queued behind it");
    mgpuComputeShaderBuildPipelineAsyncToPort(nullptr, 0, 2); // Posts 0, no crash.

    check(mgpuComputeShaderBuildPipeline(nullptr) == 0,
          "a null shader does not build");
    MGPUComputeShader *empty = mgpuCreateComputeShader();
    require(empty != nullptr, "shader creation failed");
    check(mgpuComputeShaderBuildPipeline(empty) == 0,
          "a shader without a kernel does not build");
    MGPUComputeShader *unbound = makeShader(kDouble);
    check(mgpuComputeShaderBuildPipeline(unbound) == 0,
          "a shader without bindings does not build");
    mgpuDestroyComputeShader(unbound);
    mgpuDestroyComputeShader(empty);
    mgpuDestroyComputeShader(ported);
    mgpuDestroyComputeShader(shader);
    mgpuDestroyBuffer(portedOut);
    mgpuDestroyBuffer(out);
  }

  // 3. Failed compiles: invalid WGSL (an error-scope failure under the lock)
  //    and a binding kind the kernel does not declare (the async compile's own
  //    failure status) both report 0 and write nothing; a fix then builds.
  {
    MGPUComputeShader *broken = makeShader(kNotWgsl);
    MGPUBuffer *out = makeBuffer();
    const bool bound = bind(broken, inp, out);
    check(bound && mgpuComputeShaderBuildPipeline(broken) == 0,
          "invalid WGSL makes the build report failure");
    run(broken);
    check(zeros(out), "invalid WGSL writes nothing");
    mgpuLoadKernel(broken, kDouble);
    check(mgpuComputeShaderBuildPipeline(broken) == 1,
          "a valid kernel after an invalid one builds");
    run(broken);
    check(matches(out, 2.0f, 1.0f), "and dispatches");

    MGPUComputeShader *kinds = makeShader(kDoubleReadOnly);
    MGPUBuffer *kindsOut = makeBuffer();
    const bool kindsBound = bind(kinds, inp, kindsOut, kStorage);
    check(kindsBound && mgpuComputeShaderBuildPipeline(kinds) == 0,
          "a binding kind the kernel does not declare fails the build");
    run(kinds);
    check(zeros(kindsOut), "the mismatched pipeline writes nothing");
    check(mgpuSetBufferRange(kinds, 0, inp, kReadOnlyStorage, 0, kBytes) == 0 &&
              mgpuComputeShaderBuildPipeline(kinds) == 1,
          "the matching kind builds");
    run(kinds);
    check(matches(kindsOut, 2.0f, 1.0f), "and dispatches");
    mgpuDestroyComputeShader(kinds);
    mgpuDestroyComputeShader(broken);
    mgpuDestroyBuffer(kindsOut);
    mgpuDestroyBuffer(out);
  }

  // 4. Ping-pong between two buffer sets: every dispatch accumulates into its
  //    own output, and after the first of each set the bind group is cached.
  {
    constexpr int kRounds = 50;
    MGPUComputeShader *shader = makeShader(kAccumulate);
    MGPUBuffer *inA = makeBuffer();
    MGPUBuffer *outA = makeBuffer();
    MGPUBuffer *inB = makeBuffer();
    MGPUBuffer *outB = makeBuffer();
    fill(inA, 1.0f);
    fill(inB, 1.0f, 1000.0f);
    bool bound = true;
    for (int r = 0; r < kRounds; ++r) {
      bound = bind(shader, inA, outA) && bound;
      mgpuDispatch(shader, 1, 1, 1);
      bound = bind(shader, inB, outB) && bound;
      mgpuDispatch(shader, 1, 1, 1);
    }
    mgpuDrainWorkQueue();
    check(bound && matches(outA, kRounds, 0.0f) &&
              matches(outB, kRounds, kRounds * 1000.0f),
          "ping-pong dispatches each land in their own buffers");
    const uint64_t hits = native(shader).bindGroupCacheHits();
    const uint64_t misses = native(shader).bindGroupCacheMisses();
    std::cout << "INFO ping-pong bind groups: " << hits << " cached, "
              << misses << " created" << std::endl;
    check(misses == 2 && hits == 2 * kRounds - 2,
          "ping-pong creates two bind groups and reuses them");
    mgpuDestroyComputeShader(shader);
    mgpuDestroyBuffer(outB);
    mgpuDestroyBuffer(inB);
    mgpuDestroyBuffer(outA);
    mgpuDestroyBuffer(inA);
  }

  // 5. Destroyed buffers and same-size successors: the output lands in the
  //    successor, and the destroyed buffer's data never shows. Enough rounds
  //    to evict cache entries, so released handles can be recycled.
  {
    MGPUComputeShader *shader = makeShader(kDouble);
    MGPUBuffer *a = makeBuffer();
    fill(a, 5.0f);
    MGPUBuffer *out = makeBuffer();
    bool ok = bind(shader, a, out);
    run(shader);
    ok = matches(out, 10.0f, 1.0f) && ok;
    mgpuDestroyBuffer(a);
    MGPUBuffer *b = makeBuffer();
    fill(b, 7.0f);
    ok = bind(shader, b, out) && ok;
    run(shader);
    check(ok && matches(out, 14.0f, 1.0f),
          "an input re-created after its predecessor was destroyed is read");

    MGPUBuffer *outA = makeBuffer();
    ok = bind(shader, b, outA);
    run(shader);
    ok = matches(outA, 14.0f, 1.0f) && ok;
    mgpuDestroyBuffer(outA);
    MGPUBuffer *outB = makeBuffer();
    const bool clean = zeros(outB);
    ok = bind(shader, b, outB) && ok;
    run(shader);
    check(ok && clean && matches(outB, 14.0f, 1.0f),
          "an output re-created after its predecessor was destroyed is written");

    bool churn = true;
    for (int round = 1; round <= 24; ++round) {
      MGPUBuffer *in = makeBuffer();
      MGPUBuffer *result = makeBuffer();
      fill(in, static_cast<float>(round));
      churn = bind(shader, in, result) && churn;
      run(shader);
      churn = matches(result, 2.0f * round, 1.0f) && churn;
      mgpuDestroyBuffer(result);
      mgpuDestroyBuffer(in);
    }
    check(churn, "24 rounds of destroyed and re-created buffers never alias");
    mgpuDestroyComputeShader(shader);
    mgpuDestroyBuffer(outB);
    mgpuDestroyBuffer(out);
    mgpuDestroyBuffer(b);
  }

  // 6. A new kernel, or a new binding kind on the same buffers, replaces the
  //    pipeline and every cached bind group.
  {
    MGPUComputeShader *shader = makeShader(kDouble);
    MGPUBuffer *out = makeBuffer();
    bool ok = bind(shader, inp, out);
    run(shader);
    ok = matches(out, 2.0f, 1.0f) && ok;
    mgpuLoadKernel(shader, kTriple);
    ok = bind(shader, inp, out) && ok;
    fill(out, 0.0f);
    run(shader);
    check(ok && matches(out, 3.0f, 1.0f),
          "a new kernel on the same buffers dispatches the new kernel");

    mgpuLoadKernel(shader, kDoubleReadOnly);
    ok = bind(shader, inp, out, kReadOnlyStorage);
    fill(out, 0.0f);
    run(shader);
    check(ok && matches(out, 2.0f, 1.0f),
          "a binding-kind change on the same buffers rebuilds the bind group");

    mgpuLoadKernel(shader, kTriple);
    ok = bind(shader, inp, out, kStorage);
    fill(out, 0.0f);
    run(shader);
    check(ok && matches(out, 3.0f, 1.0f), "and changing it back rebuilds again");
    mgpuDestroyComputeShader(shader);
    mgpuDestroyBuffer(out);
  }

  // 7. The GPU mutex stays free while a large kernel compiles: another
  //    thread's mutex-taking call (mgpuHasKernel) does not wait for it.
  {
    const std::string kernel = largeKernel(1500);
    MGPUComputeShader *shader = makeShader(kernel.c_str());
    MGPUComputeShader *probe = makeShader(kDouble);
    MGPUBuffer *out = makeBuffer();
    const bool bound = bind(shader, inp, out);
    mgpuDrainWorkQueue();
    std::atomic<bool> building{true};
    double longestWait = 0.0;
    std::thread prober([&] {
      while (building.load(std::memory_order_acquire)) {
        const auto start = std::chrono::steady_clock::now();
        mgpuHasKernel(probe);
        longestWait = std::max(longestWait, millisSince(start));
      }
    });
    const auto start = std::chrono::steady_clock::now();
    const int built = mgpuComputeShaderBuildPipeline(shader);
    const double buildMs = millisSince(start);
    building.store(false, std::memory_order_release);
    prober.join();
    std::cout << "INFO large kernel build " << buildMs
              << " ms; longest GPU-mutex wait meanwhile " << longestWait
              << " ms" << std::endl;
    check(bound && built == 1, "a large kernel builds");
    if (buildMs >= 200.0) {
      check(longestWait < buildMs / 2.0,
            "the GPU mutex stays free while a pipeline compiles");
    } else {
      std::cout << "INFO compile too fast to time the GPU mutex against"
                << std::endl;
    }
    mgpuDestroyComputeShader(probe);
    mgpuDestroyComputeShader(shader);
    mgpuDestroyBuffer(out);
  }

  mgpuDestroyBuffer(inp);
  mgpuDestroyContext();
  std::cout << (failures ? "compute_pipeline_async_test FAILED"
                         : "compute_pipeline_async_test passed")
            << std::endl;
  return failures ? 1 : 0;
}

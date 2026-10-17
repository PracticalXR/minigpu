#include "../include/minigpu.h"

#include <climits>
#include <cstdlib>
#include <iostream>

// One bad shader must never cost another shader its results. Fire-and-forget
// dispatches share one encoder until the next flush, so a single invalid
// command recorded there voids every dispatch in it; and a binding slot the
// tables cannot hold must not close (or crash) the shader that received it.

namespace {

constexpr int kF32 = 1; // mgpuCreateBuffer data type code
constexpr int kCount = 64;
constexpr int kBytes = kCount * static_cast<int>(sizeof(float));
constexpr int kStorage = 0;         // mgpuSetBufferRange kinds
constexpr int kReadOnlyStorage = 1;

// out[i] = inp[i] * 2 + 1, so a dispatch that never ran leaves zeros.
const char *kDouble = R"(
@group(0) @binding(0) var<storage, read_write> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i < arrayLength(&inp)) { out[i] = inp[i] * 2.0 + 1.0; }
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

const char *kTripleReadOnly = R"(
@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i < arrayLength(&inp)) { out[i] = inp[i] * 3.0 + 1.0; }
}
)";

// Bare `var<storage>` is read-only; the commented-out declarations must not
// count.
const char *kTripleBareStorage = R"(
// @group(0) @binding(0) var<storage, read_write> inp: array<f32>;
/* @group(0) @binding(0) /* nested */ var<storage, read_write> inp: array<f32>; */
@group(0) @binding(0)
var<storage> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i < arrayLength(&inp)) { out[i] = inp[i] * 3.0 + 1.0; }
}
)";

const char *kNotWgsl = "fn main( { this is not WGSL";

void require(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(1);
  }
}

void fill(MGPUBuffer *buffer, float scale) {
  float data[kCount];
  for (int i = 0; i < kCount; ++i) data[i] = scale * static_cast<float>(i);
  mgpuWriteFloat(buffer, data, sizeof(data));
}

// Zeroed explicitly: "never written" must not depend on allocation details.
MGPUBuffer *makeBuffer() {
  MGPUBuffer *buffer = mgpuCreateBuffer(kBytes, kF32);
  require(buffer != nullptr, "buffer creation failed");
  fill(buffer, 0.0f);
  return buffer;
}

// True when out[i] == i * scale * mul + 1 for every i; mul 0 means "never
// written" (all zeros). Small integers, so the GPU's result is exact.
bool holds(MGPUBuffer *out, float scale, float mul) {
  float data[kCount] = {};
  mgpuReadSync(out, data, sizeof(data), 0);
  for (int i = 0; i < kCount; ++i) {
    const float want =
        mul == 0.0f ? 0.0f : scale * static_cast<float>(i) * mul + 1.0f;
    if (data[i] != want) return false;
  }
  return true;
}

MGPUComputeShader *makeShader(const char *kernel) {
  MGPUComputeShader *shader = mgpuCreateComputeShader();
  require(shader != nullptr, "shader creation failed");
  mgpuLoadKernel(shader, kernel);
  return shader;
}

// Dispatches are queued; drain so they are recorded before the read flushes.
void dispatchAll(MGPUComputeShader *first, int firstX,
                 MGPUComputeShader *second) {
  mgpuDispatch(first, firstX, 1, 1);
  mgpuDispatch(second, 1, 1, 1);
  mgpuDrainWorkQueue();
}

} // namespace

int main() {
  mgpuInitializeContext();

  MGPUBuffer *inp = makeBuffer();
  fill(inp, 1.0f);

  // An out-of-range slot is ignored on both setters: the queued one used to
  // close the shader, the direct one threw out of the C export.
  MGPUComputeShader *good = makeShader(kDouble);
  MGPUBuffer *goodOut = makeBuffer();
  mgpuSetBufferFire(good, INT_MAX, inp);
  mgpuDrainWorkQueue();
  mgpuSetBuffer(good, INT_MAX, inp);
  mgpuSetBuffer(good, 0, inp);
  mgpuSetBuffer(good, 1, goodOut);
  mgpuDispatch(good, 1, 1, 1);
  mgpuDrainWorkQueue();
  require(holds(goodOut, 1.0f, 2.0f),
          "an out-of-range binding slot broke the shader");

  // Over-limit workgroup count, batched ahead of a good dispatch.
  MGPUComputeShader *wide = makeShader(kDouble);
  MGPUBuffer *wideOut = makeBuffer();
  mgpuSetBuffer(wide, 0, inp);
  mgpuSetBuffer(wide, 1, wideOut);
  fill(goodOut, 0.0f);
  dispatchAll(wide, 70000, good);
  require(holds(goodOut, 1.0f, 2.0f),
          "an over-limit dispatch voided another shader's batched dispatch");

  // Kernel that does not compile, batched ahead of a good dispatch.
  MGPUComputeShader *broken = makeShader(kNotWgsl);
  MGPUBuffer *brokenOut = makeBuffer();
  mgpuSetBuffer(broken, 0, inp);
  mgpuSetBuffer(broken, 1, brokenOut);
  fill(goodOut, 0.0f);
  dispatchAll(broken, 1, good);
  require(holds(goodOut, 1.0f, 2.0f),
          "an invalid kernel voided another shader's batched dispatch");
  // The failure is remembered, not retried into the batch.
  fill(goodOut, 0.0f);
  dispatchAll(broken, 1, good);
  require(holds(goodOut, 1.0f, 2.0f),
          "a known-bad kernel voided a later batch");
  require(holds(brokenOut, 1.0f, 0.0f), "an invalid kernel wrote output");

  // A binding kind the kernel does not declare fails pipeline creation; a
  // matching kind rebuilds the pipeline on the module already parsed; a new
  // kernel rebuilds the module.
  MGPUComputeShader *kinds = makeShader(kDoubleReadOnly);
  MGPUBuffer *kindsOut = makeBuffer();
  require(mgpuSetBufferRange(kinds, 0, inp, kStorage, 0, kBytes) == 0,
          "read-write bind of slot 0 rejected");
  require(mgpuSetBufferRange(kinds, 1, kindsOut, kStorage, 0, kBytes) == 0,
          "bind of slot 1 rejected");
  fill(goodOut, 0.0f);
  dispatchAll(kinds, 1, good);
  require(holds(goodOut, 1.0f, 2.0f),
          "a layout mismatch voided another shader's batched dispatch");
  require(holds(kindsOut, 1.0f, 0.0f), "a mismatched pipeline wrote output");

  require(mgpuSetBufferRange(kinds, 0, inp, kReadOnlyStorage, 0, kBytes) == 0,
          "read-only bind of slot 0 rejected");
  mgpuDispatch(kinds, 1, 1, 1);
  mgpuDrainWorkQueue();
  require(holds(kindsOut, 1.0f, 2.0f),
          "a matching binding kind did not recover the shader");

  mgpuLoadKernel(kinds, kTripleReadOnly);
  mgpuDispatch(kinds, 1, 1, 1);
  mgpuDrainWorkQueue();
  require(holds(kindsOut, 1.0f, 3.0f), "a new kernel reused the old module");

  // A buffer bound without a kind takes the access its kernel declares, on
  // both setters, and follows a new kernel's declarations.
  MGPUComputeShader *plain = makeShader(kDoubleReadOnly);
  MGPUBuffer *plainOut = makeBuffer();
  mgpuSetBuffer(plain, 0, inp);
  mgpuSetBufferFire(plain, 1, plainOut);
  mgpuDispatch(plain, 1, 1, 1);
  mgpuDrainWorkQueue();
  require(holds(plainOut, 1.0f, 2.0f),
          "a plain bind of a read-only input did not dispatch");
  mgpuLoadKernel(plain, kTripleBareStorage);
  mgpuDispatch(plain, 1, 1, 1);
  mgpuDrainWorkQueue();
  require(holds(plainOut, 1.0f, 3.0f),
          "a plain bind of a bare var<storage> input did not dispatch");
  mgpuLoadKernel(plain, kDouble);
  mgpuDispatch(plain, 1, 1, 1);
  mgpuDrainWorkQueue();
  require(holds(plainOut, 1.0f, 2.0f),
          "a plain bind did not follow a read_write declaration");

  mgpuDestroyComputeShader(plain);
  mgpuDestroyBuffer(plainOut);
  mgpuDestroyComputeShader(kinds);
  mgpuDestroyComputeShader(broken);
  mgpuDestroyComputeShader(wide);
  mgpuDestroyComputeShader(good);
  mgpuDestroyBuffer(kindsOut);
  mgpuDestroyBuffer(brokenOut);
  mgpuDestroyBuffer(wideOut);
  mgpuDestroyBuffer(goodOut);
  mgpuDestroyBuffer(inp);
  mgpuDestroyContext();
  std::cout << "compute_guard_test passed" << std::endl;
  return 0;
}

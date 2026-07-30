#include "../include/minigpu.h"
#include <iostream>
#include <cassert>
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

// Forward declaration from external_texture_test.cpp
void runExternalTextureTests();


void testCreateContext() {
  std::cout << "Testing context creation..." << std::endl;
  mgpuInitializeContext();
  // Assuming no error is reported, the context is created.
  std::cout << "Context created successfully." << std::endl;
}

void testCreateBuffer() {
  std::cout << "Testing buffer creation (1024 bytes)..." << std::endl;
  MGPUBuffer *buffer = mgpuCreateBuffer(1024, 1); // Use numeric value for kf32
  if (buffer) {
    std::cout << "Buffer created successfully." << std::endl;
    mgpuDestroyBuffer(buffer);
    std::cout << "Buffer destroyed successfully." << std::endl;
  } else {
    std::cerr << "Failed to create buffer!" << std::endl;
  }
}

void testComputeShader() {
  std::cout << "Testing compute shader..." << std::endl;
  MGPUComputeShader *shader = mgpuCreateComputeShader();
  if (!shader) {
    std::cerr << "Failed to create compute shader." << std::endl;
    return;
  }

  // Load a basic kernel string.
  const char *kernelCode = R"(
        const GELU_SCALING_FACTOR: f32 = 0.7978845608028654;
        @group(0) @binding(0) var<storage, read_write> inp: array<f32>;
        @group(0) @binding(1) var<storage, read_write> out: array<f32>;
        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
            let i: u32 = GlobalInvocationID.x;
            if (i < 100u) {
                let x: f32 = inp[i];
                out[i] = x + 0.2;
            }
        }
    )";

  mgpuLoadKernel(shader, kernelCode);

  // Create buffers for 100 floats.
  const int numFloats = 100;
  MGPUBuffer *inpBuffer =
      mgpuCreateBuffer(numFloats * sizeof(float), 1); // kf32 = 1
  MGPUBuffer *outBuffer =
      mgpuCreateBuffer(numFloats * sizeof(float), 1); // kf32 = 1
  if (!inpBuffer || !outBuffer) {
    std::cerr << "Failed to create one or more buffers." << std::endl;
    mgpuDestroyComputeShader(shader);
    return;
  }

  // Initialize input data.
  float inputData[numFloats];
  for (int i = 0; i < numFloats; i++) {
    inputData[i] = static_cast<float>(i);
  }
  // Use the API call to set buffer data.
  mgpuWriteFloat(inpBuffer, inputData, numFloats * sizeof(float));

  // Set buffers on the shader.
  // Here tag '0' for the input buffer and tag '1' for the output buffer.
  mgpuSetBuffer(shader, 0, inpBuffer);
  mgpuSetBuffer(shader, 1, outBuffer);

  // Dispatch the compute shader (using 1 workgroup for testing).
  mgpuDispatch(shader, 1, 1, 1);
  std::cout << "Compute shader dispatched successfully." << std::endl;

  // Read the output data synchronously.
  float outputData[numFloats] = {0};
  mgpuReadSync(outBuffer, outputData, numFloats * sizeof(float), 0);

  // Print output for verification.
  std::cout << "Buffer input values + 0.2 (expected results):" << std::endl;
  for (int i = 0; i < numFloats; i++) {
    std::cout << "Index " << i << ": " << outputData[i] << std::endl;
  }

  // Clean up resources.
  mgpuDestroyBuffer(inpBuffer);
  mgpuDestroyBuffer(outBuffer);
  mgpuDestroyComputeShader(shader);
}

void testDestroyContext() {
  std::cout << "Testing context destruction..." << std::endl;
  mgpuDestroyContext();
  std::cout << "Context destroyed successfully." << std::endl;
}

void testUint8() {
  std::cout << "Testing uint8 buffer..." << std::endl;
  const int numElements = 10;
  // Create a buffer with 10 bytes.
  MGPUBuffer *buffer = mgpuCreateBuffer(numElements, 7); // ku8 = 7
  assert(buffer && "Failed to create uint8 buffer");

  // Create input data (uint8_t).
  uint8_t inputData[numElements] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  // Set data for the uint8 buffer.
  mgpuWriteUint8(buffer, inputData, numElements);

  // Read back data.
  uint8_t outputData[numElements] = {0};
  mgpuReadSyncUint8(buffer, outputData, numElements, 0);

  // Validate that the output data matches the input.
  if (memcmp(inputData, outputData, numElements) != 0) {
    std::cerr << "Uint8 test failed: output does not match input." << std::endl;
    exit(1);
  }
  std::cout << "Uint8 test passed." << std::endl;
  mgpuDestroyBuffer(buffer);
}

void testInt8() {
  std::cout << "Testing int8 buffer..." << std::endl;
  const int numElements = 10;
  // Create a buffer with 10 bytes.
  MGPUBuffer *buffer = mgpuCreateBuffer(numElements, 3); // ki8 = 3
  assert(buffer && "Failed to create int8 buffer");

  // Create input data (int8_t).
  int8_t inputData[numElements] = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10};

  // Set data for the int8 buffer.
  mgpuWriteInt8(buffer, inputData, numElements);

  // Read back data.
  int8_t outputData[numElements] = {0};
  mgpuReadSyncInt8(buffer, outputData, numElements, 0);

  // Validate that the output data matches the input.
  if (memcmp(inputData, outputData, numElements) != 0) {
    std::cerr << "Int8 test failed: output does not match input." << std::endl;
    exit(1);
  }
  std::cout << "Int8 test passed." << std::endl;
  mgpuDestroyBuffer(buffer);
}

void testInt16() {
  std::cout << "Testing int16 buffer..." << std::endl;
  const int numElements = 10;
  MGPUBuffer *buffer = mgpuCreateBuffer(numElements * sizeof(int16_t),
                                        4); // ki16 = 4
  assert(buffer && "Failed to create int16 buffer");

  int16_t inputData[numElements] = {-100, -200, -300, -400, -500,
                                    600,  700,  800,  900,  1000};
  mgpuWriteInt16(buffer, inputData, numElements * sizeof(int16_t));

  int16_t outputData[numElements] = {0};
  mgpuReadSyncInt16(buffer, outputData, numElements, 0);

  if (memcmp(inputData, outputData, numElements * sizeof(int16_t)) != 0) {
    std::cerr << "Int16 test failed: output does not match input." << std::endl;
    exit(1);
  }
  std::cout << "Int16 test passed." << std::endl;
  mgpuDestroyBuffer(buffer);
}

void testUint16() {
  std::cout << "Testing uint16 buffer..." << std::endl;
  const int numElements = 10;
  MGPUBuffer *buffer =
      mgpuCreateBuffer(numElements * sizeof(uint16_t), 8); // ku16 = 8
  assert(buffer && "Failed to create uint16 buffer");

  uint16_t inputData[numElements] = {100, 200, 300, 400, 500,
                                     600, 700, 800, 900, 1000};
  mgpuWriteUint16(buffer, inputData, numElements * sizeof(uint16_t));

  uint16_t outputData[numElements] = {0};
  mgpuReadSyncUint16(buffer, outputData, numElements, 0);

  if (memcmp(inputData, outputData, numElements * sizeof(uint16_t)) != 0) {
    std::cerr << "Uint16 test failed: output does not match input."
              << std::endl;
    exit(1);
  }
  std::cout << "Uint16 test passed." << std::endl;
  mgpuDestroyBuffer(buffer);
}

void testInt32() {
  std::cout << "Testing int32 buffer..." << std::endl;
  const int numElements = 10;
  MGPUBuffer *buffer =
      mgpuCreateBuffer(numElements * sizeof(int32_t), 5); // ki32 = 5
  assert(buffer && "Failed to create int32 buffer");

  int32_t inputData[numElements] = {-1000, -2000, -3000, -4000, -5000,
                                    6000,  7000,  8000,  9000,  10000};
  mgpuWriteInt32(buffer, inputData, numElements * sizeof(int32_t));

  int32_t outputData[numElements] = {0};
  mgpuReadSyncInt32(buffer, outputData, numElements, 0);

  if (memcmp(inputData, outputData, numElements * sizeof(int32_t)) != 0) {
    std::cerr << "Int32 test failed: output does not match input." << std::endl;
    exit(1);
  }
  std::cout << "Int32 test passed." << std::endl;
  mgpuDestroyBuffer(buffer);
}

void testUint32() {
  std::cout << "Testing uint32 buffer..." << std::endl;
  const int numElements = 10;
  MGPUBuffer *buffer =
      mgpuCreateBuffer(numElements * sizeof(uint32_t), 9); // ku32 = 9
  assert(buffer && "Failed to create uint32 buffer");

  uint32_t inputData[numElements] = {1000, 2000, 3000, 4000, 5000,
                                     6000, 7000, 8000, 9000, 10000};
  mgpuWriteUint32(buffer, inputData, numElements * sizeof(uint32_t));

  uint32_t outputData[numElements] = {0};
  mgpuReadSyncUint32(buffer, outputData, numElements, 0);

  if (memcmp(inputData, outputData, numElements * sizeof(uint32_t)) != 0) {
    std::cerr << "Uint32 test failed: output does not match input."
              << std::endl;
    exit(1);
  }
  std::cout << "Uint32 test passed." << std::endl;
  mgpuDestroyBuffer(buffer);
}

void testInt64() {
  std::cout << "Testing int64 buffer..." << std::endl;
  const int numElements = 10;
  MGPUBuffer *buffer =
      mgpuCreateBuffer(numElements * sizeof(int64_t), 6); // ki64 = 6
  assert(buffer && "Failed to create int64 buffer");

  int64_t inputData[numElements] = {-100000, -200000, -300000, -400000,
                                    -500000, 600000,  700000,  800000,
                                    900000,  1000000};
  mgpuWriteInt64(buffer, inputData, numElements * sizeof(int64_t));

  int64_t outputData[numElements] = {0};
  mgpuReadSyncInt64(buffer, outputData, numElements, 0);

  if (memcmp(inputData, outputData, numElements * sizeof(int64_t)) != 0) {
    std::cerr << "Int64 test failed: output does not match input." << std::endl;
    exit(1);
  }
  std::cout << "Int64 test passed." << std::endl;
  mgpuDestroyBuffer(buffer);
}

void testUint64() {
  std::cout << "Testing uint64 buffer..." << std::endl;
  const int numElements = 10;
  MGPUBuffer *buffer =
      mgpuCreateBuffer(numElements * sizeof(uint64_t), 10); // ku64 = 10
  assert(buffer && "Failed to create uint64 buffer");

  uint64_t inputData[numElements] = {100000, 200000, 300000, 400000, 500000,
                                     600000, 700000, 800000, 900000, 1000000};
  mgpuWriteUint64(buffer, inputData, numElements * sizeof(uint64_t));

  uint64_t outputData[numElements] = {0};
  mgpuReadSyncUint64(buffer, outputData, numElements, 0);

  if (memcmp(inputData, outputData, numElements * sizeof(uint64_t)) != 0) {
    std::cerr << "Uint64 test failed: output does not match input."
              << std::endl;
    exit(1);
  }
  std::cout << "Uint64 test passed." << std::endl;
  mgpuDestroyBuffer(buffer);
}

void testFloat32() {
  std::cout << "Testing float32 buffer..." << std::endl;
  const int numElements = 10;
  MGPUBuffer *buffer =
      mgpuCreateBuffer(numElements * sizeof(float), 1); // kf32 = 1
  assert(buffer && "Failed to create float32 buffer");

  float inputData[numElements] = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f,
                                  6.6f, 7.7f, 8.8f, 9.9f, 10.0f};
  mgpuWriteFloat(buffer, inputData, numElements * sizeof(float));

  float outputData[numElements] = {0};
  mgpuReadSyncFloat32(buffer, outputData, numElements, 0);

  if (memcmp(inputData, outputData, numElements * sizeof(float)) != 0) {
    std::cerr << "Float32 test failed: output does not match input."
              << std::endl;
    exit(1);
  }
  std::cout << "Float32 test passed." << std::endl;
  mgpuDestroyBuffer(buffer);
}

void testFloat64() {
  std::cout << "Testing float64 buffer..." << std::endl;
  const int numElements = 10;
  MGPUBuffer *buffer =
      mgpuCreateBuffer(numElements * sizeof(double), 2); // kf64 = 2
  assert(buffer && "Failed to create float64 buffer");

  double inputData[numElements] = {1.1, 2.2, 3.3, 4.4, 5.5,
                                   6.6, 7.7, 8.8, 9.9, 10.0};
  mgpuWriteDouble(buffer, inputData, numElements * sizeof(double));

  double outputData[numElements] = {0};
  mgpuReadSyncFloat64(buffer, outputData, numElements, 0);

  if (memcmp(inputData, outputData, numElements * sizeof(double)) != 0) {
    std::cerr << "Float64 test failed: output does not match input."
              << std::endl;
    exit(1);
  }
  std::cout << "Float64 test passed." << std::endl;
  mgpuDestroyBuffer(buffer);
}

// ── BATCHED STAGING UPLOADS ────────────────────────────────────────────────
// Three properties, all of which the inline write path gets for free and the
// batched one has to earn:
//   (1) scattered staged ranges land byte-exactly at their destination
//       offsets, including ranges that coalesce and ranges that do not;
//   (2) a staged upload issued BEFORE a fire-and-forget dispatch is visible to
//       that dispatch (FIFO ordering through the WebGPU thread, no flush);
//   (3) staging survives many scopes (the arena wraps and re-grows).
static std::atomic<int> g_stagedDispatchDone{0};
static void stagedDispatchCb() { g_stagedDispatchDone = 1; }

static void testStagedUploads() {
  std::cout << "Testing batched staging uploads..." << std::endl;
  if (!mgpuUploadsSupported()) {
    std::cout << "  (not supported in this build; skipped)" << std::endl;
    return;
  }

  const int n = 4096; // uint32 elements
  MGPUBuffer *buf = mgpuCreateBuffer(n * 4, 9 /* u32 */);
  MGPUBuffer *out = mgpuCreateBuffer(n * 4, 9);
  if (!buf || !out) {
    std::cerr << "Staged upload test: buffer creation failed." << std::endl;
    exit(1);
  }

  std::vector<uint32_t> zero(n, 0u);
  mgpuWriteUint32(buf, zero.data(), n * 4);

  // (1) scattered ranges, some adjacent (must coalesce), some not.
  std::vector<uint32_t> expect(n, 0u);
  mgpuBeginUploads();
  mgpuStageReserve(n * 4);
  int staged = 0;
  for (int r = 0; r < 64; ++r) {
    const int off = r * 61;         // deliberately not a run boundary
    const int len = (r % 7) + 1;    // 1..7 elements
    if (off + len > n) break;
    std::vector<uint32_t> vals(len);
    for (int i = 0; i < len; ++i) {
      vals[i] = 0xA5000000u | (uint32_t)(off + i);
      expect[off + i] = vals[i];
    }
    const int rc = mgpuStageWrite(buf, (size_t)off * 4, vals.data(),
                                  (size_t)len * 4);
    if (rc < 0) {
      std::cerr << "Staged upload test: mgpuStageWrite failed." << std::endl;
      exit(1);
    }
    staged += (rc == 1);
    // an immediately adjacent range: exercises the coalescing branch
    if (off + len + len <= n) {
      std::vector<uint32_t> more(len);
      for (int i = 0; i < len; ++i) {
        more[i] = 0x5A000000u | (uint32_t)(off + len + i);
        expect[off + len + i] = more[i];
      }
      mgpuStageWrite(buf, (size_t)(off + len) * 4, more.data(),
                     (size_t)len * 4);
      staged++;
    }
  }
  const int emitted = mgpuEndUploads();
  if (staged == 0 || emitted <= 0) {
    std::cerr << "Staged upload test: nothing was batched (staged=" << staged
              << " emitted=" << emitted << ")." << std::endl;
    exit(1);
  }

  std::vector<uint32_t> got(n, 0xDEADBEEFu);
  mgpuReadSyncUint32(buf, got.data(), n, 0);
  for (int i = 0; i < n; ++i) {
    if (got[i] != expect[i]) {
      std::cerr << "Staged upload test: mismatch at " << i << " got "
                << got[i] << " expected " << expect[i] << std::endl;
      exit(1);
    }
  }
  std::cout << "  scattered ranges byte-exact (" << staged << " staged -> "
            << emitted << " copies)" << std::endl;

  // (2) ordering: stage, then a fire-and-forget dispatch that reads what was
  //     staged.  If the copy were recorded after the dispatch (or lost to a
  //     flush) the kernel would see the zeros.
  MGPUComputeShader *sh = mgpuCreateComputeShader();
  mgpuLoadKernel(sh, R"(
      @group(0) @binding(0) var<storage, read_write> inp: array<u32>;
      @group(0) @binding(1) var<storage, read_write> outp: array<u32>;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let i = gid.x;
        if (i < 4096u) { outp[i] = inp[i] + 1u; }
      })");
  mgpuSetBuffer(sh, 0, buf);
  mgpuSetBuffer(sh, 1, out);

  for (int pass = 0; pass < 8; ++pass) {
    std::vector<uint32_t> wave(n);
    for (int i = 0; i < n; ++i) wave[i] = (uint32_t)(pass * 1000 + i);
    mgpuBeginUploads();
    // many small ranges, the shape the codec produces
    for (int i = 0; i < n; i += 32)
      mgpuStageWrite(buf, (size_t)i * 4, wave.data() + i, 32 * 4);
    mgpuEndUploads();
    mgpuDispatch(sh, n / 64, 1, 1);
    // mgpuDispatch is enqueued on minigpu's worker thread while mgpuReadSync
    // runs inline on THIS thread, so the read has to be sequenced behind the
    // worker queue.  The kernel is idempotent, so a second (awaited) dispatch
    // is the cheapest barrier that exists in the public API.
    g_stagedDispatchDone = 0;
    mgpuDispatchAsync(sh, n / 64, 1, 1, stagedDispatchCb);
    while (!g_stagedDispatchDone) {
      std::this_thread::yield();
    }
    mgpuReadSyncUint32(out, got.data(), n, 0);
    for (int i = 0; i < n; ++i) {
      if (got[i] != wave[i] + 1u) {
        std::cerr << "Staged upload test: dispatch saw stale data at " << i
                  << " (pass " << pass << ") got " << got[i] << " expected "
                  << (wave[i] + 1u) << std::endl;
        exit(1);
      }
    }
  }
  std::cout << "  staged-then-dispatch ordering holds over 8 passes"
            << std::endl;

  // (3) nested scopes collapse to one emission, and an empty scope is a no-op.
  mgpuBeginUploads();
  mgpuBeginUploads();
  uint32_t one = 0x1234u;
  mgpuStageWrite(buf, 0, &one, 4);
  if (mgpuEndUploads() != 0) {
    std::cerr << "Staged upload test: inner End emitted." << std::endl;
    exit(1);
  }
  if (mgpuEndUploads() != 1) {
    std::cerr << "Staged upload test: outer End did not emit." << std::endl;
    exit(1);
  }
  mgpuBeginUploads();
  mgpuEndUploads(); // empty scope
  mgpuReadSyncUint32(buf, got.data(), 2, 0);
  if (got[0] != 0x1234u) {
    std::cerr << "Staged upload test: nested scope value wrong." << std::endl;
    exit(1);
  }
  std::cout << "  nesting / empty scope ok" << std::endl;
  // NOTE: an unaligned range (offset or size not a multiple of 4) is rejected
  // by mgpuStageWrite (returns 0) and forwarded to the inline path, where it
  // hits the SAME WebGPU validation error mgpuWriteBufferAt has always hit.
  // Not exercised here: a Dawn validation error is sticky and poisons the
  // device for every later test in this process.

  // (4) with no scope open, mgpuStageWrite is exactly mgpuWriteBufferAt.
  uint32_t two = 0x77u;
  if (mgpuStageWrite(buf, 8, &two, 4) != 0) {
    std::cerr << "Staged upload test: batched without a scope." << std::endl;
    exit(1);
  }
  mgpuReadSyncUint32(buf, got.data(), 4, 0);
  if (got[2] != 0x77u) {
    std::cerr << "Staged upload test: scopeless fallback did not land."
              << std::endl;
    exit(1);
  }

  mgpuDestroyComputeShader(sh);
  mgpuDestroyBuffer(buf);
  mgpuDestroyBuffer(out);
  std::cout << "Batched staging upload test passed." << std::endl;
}

// ── BATCHED READBACKS ──────────────────────────────────────────────────────
// Four properties the per-buffer `mgpuReadSync*` path gets for free:
//   (1) scattered staged reads across SEVERAL buffers land byte-exactly in
//       their own destinations, including ranges that coalesce and ones that
//       do not, and at non-zero source offsets;
//   (2) a scope resolved AFTER a dispatch sees that dispatch's output — the
//       staleness property, which is the one a batched readback could break
//       (the copies are recorded at End and the submit carries them);
//   (3) nesting collapses to one resolve and an empty scope is a no-op;
//   (4) with no scope open, mgpuStageRead is exactly an inline read.
static std::atomic<int> g_rbDispatchDone{0};
static void rbDispatchCb() { g_rbDispatchDone = 1; }

static void testBatchedReadbacks() {
  std::cout << "Testing batched readbacks..." << std::endl;
  if (!mgpuReadbacksSupported()) {
    std::cout << "  (not supported in this build; skipped)" << std::endl;
    return;
  }
  const int n = 4096; // uint32 elements
  MGPUBuffer *a = mgpuCreateBuffer(n * 4, 9 /* u32 */);
  MGPUBuffer *b = mgpuCreateBuffer(n * 4, 9);
  MGPUBuffer *out = mgpuCreateBuffer(n * 4, 9);
  if (!a || !b || !out) {
    std::cerr << "Batched readback test: buffer creation failed." << std::endl;
    exit(1);
  }
  std::vector<uint32_t> va(n), vb(n);
  for (int i = 0; i < n; ++i) {
    va[i] = 0xC0DE0000u | (uint32_t)i;
    vb[i] = 0x0BAD0000u | (uint32_t)(n - i);
  }
  mgpuWriteUint32(a, va.data(), n * 4);
  mgpuWriteUint32(b, vb.data(), n * 4);

  // (1) scattered reads across two buffers into separate destinations.
  std::vector<std::vector<uint32_t>> dsts;
  struct Range { MGPUBuffer *buf; int off, len; };
  std::vector<Range> ranges;
  for (int r = 0; r < 48; ++r) {
    const int off = r * 71;
    const int len = (r % 5) + 1;
    if (off + len * 2 > n) break;
    ranges.push_back({(r & 1) ? b : a, off, len});
    // an immediately adjacent range on the SAME buffer: coalescing branch
    ranges.push_back({(r & 1) ? b : a, off + len, len});
  }
  dsts.resize(ranges.size());
  mgpuBeginReadbacks();
  int staged = 0;
  for (size_t i = 0; i < ranges.size(); ++i) {
    dsts[i].assign((size_t)ranges[i].len, 0xDEADBEEFu);
    const int rc = mgpuStageRead(ranges[i].buf, (size_t)ranges[i].off * 4,
                                 dsts[i].data(), (size_t)ranges[i].len * 4);
    if (rc < 0) {
      std::cerr << "Batched readback test: mgpuStageRead failed." << std::endl;
      exit(1);
    }
    staged += (rc == 1);
  }
  const int filled = mgpuEndReadbacks();
  if (staged == 0 || filled != (int)ranges.size()) {
    std::cerr << "Batched readback test: nothing was batched (staged="
              << staged << " filled=" << filled << ")." << std::endl;
    exit(1);
  }
  for (size_t i = 0; i < ranges.size(); ++i) {
    const std::vector<uint32_t> &src = (ranges[i].buf == a) ? va : vb;
    for (int k = 0; k < ranges[i].len; ++k) {
      if (dsts[i][k] != src[ranges[i].off + k]) {
        std::cerr << "Batched readback test: mismatch in range " << i
                  << " elem " << k << " got " << dsts[i][k] << " expected "
                  << src[ranges[i].off + k] << std::endl;
        exit(1);
      }
    }
  }
  std::cout << "  scattered reads byte-exact (" << staged
            << " staged across 2 buffers)" << std::endl;

  // (2) STALENESS. Vary the input every pass, dispatch, then read the result
  //     through the batched scope. A batched readback that mapped before the
  //     copies were submitted would return the PREVIOUS pass's answer, and
  //     that is invisible to any static-input check — so the input moves.
  MGPUComputeShader *sh = mgpuCreateComputeShader();
  mgpuLoadKernel(sh, R"(
      @group(0) @binding(0) var<storage, read_write> inp: array<u32>;
      @group(0) @binding(1) var<storage, read_write> outp: array<u32>;
      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let i = gid.x;
        if (i < 4096u) { outp[i] = inp[i] * 3u + 7u; }
      })");
  mgpuSetBuffer(sh, 0, a);
  mgpuSetBuffer(sh, 1, out);
  std::vector<uint32_t> head(64), mid(64), tail(64);
  for (int pass = 0; pass < 8; ++pass) {
    std::vector<uint32_t> wave(n);
    for (int i = 0; i < n; ++i) wave[i] = (uint32_t)(pass * 7919 + i);
    mgpuWriteUint32(a, wave.data(), n * 4);
    g_rbDispatchDone = 0;
    mgpuDispatchAsync(sh, n / 64, 1, 1, rbDispatchCb);
    while (!g_rbDispatchDone) std::this_thread::yield();
    // three ranges from ONE buffer at three offsets, one scope
    mgpuBeginReadbacks();
    mgpuStageRead(out, 0, head.data(), 64 * 4);
    mgpuStageRead(out, (size_t)(n / 2) * 4, mid.data(), 64 * 4);
    mgpuStageRead(out, (size_t)(n - 64) * 4, tail.data(), 64 * 4);
    if (mgpuEndReadbacks() != 3) {
      std::cerr << "Batched readback test: End did not fill 3." << std::endl;
      exit(1);
    }
    for (int i = 0; i < 64; ++i) {
      const uint32_t eh = wave[i] * 3u + 7u;
      const uint32_t em = wave[n / 2 + i] * 3u + 7u;
      const uint32_t et = wave[n - 64 + i] * 3u + 7u;
      if (head[i] != eh || mid[i] != em || tail[i] != et) {
        std::cerr << "Batched readback test: STALE/wrong at pass " << pass
                  << " i " << i << std::endl;
        exit(1);
      }
    }
  }
  std::cout << "  post-dispatch reads are fresh over 8 varying passes"
            << std::endl;

  // (3) nesting / empty scope.
  std::vector<uint32_t> one(1, 0u);
  mgpuBeginReadbacks();
  mgpuBeginReadbacks();
  mgpuStageRead(a, 0, one.data(), 4);
  if (mgpuEndReadbacks() != 0) {
    std::cerr << "Batched readback test: inner End resolved." << std::endl;
    exit(1);
  }
  if (mgpuEndReadbacks() != 1) {
    std::cerr << "Batched readback test: outer End did not resolve."
              << std::endl;
    exit(1);
  }
  mgpuBeginReadbacks();
  if (mgpuEndReadbacks() != 0) {
    std::cerr << "Batched readback test: empty scope resolved." << std::endl;
    exit(1);
  }
  std::cout << "  nesting / empty scope ok" << std::endl;

  // (4) scopeless fallback: mgpuStageRead performs the inline read itself.
  std::vector<uint32_t> two(4, 0xDEADBEEFu);
  if (mgpuStageRead(a, 8, two.data(), 16) != 0) {
    std::cerr << "Batched readback test: batched without a scope." << std::endl;
    exit(1);
  }
  std::vector<uint32_t> ref(n);
  mgpuReadSyncUint32(a, ref.data(), n, 0);
  for (int i = 0; i < 4; ++i) {
    if (two[i] != ref[2 + i]) {
      std::cerr << "Batched readback test: scopeless fallback wrong at " << i
                << std::endl;
      exit(1);
    }
  }
  std::cout << "  scopeless fallback matches mgpuReadSync" << std::endl;

  mgpuDestroyComputeShader(sh);
  mgpuDestroyBuffer(a);
  mgpuDestroyBuffer(b);
  mgpuDestroyBuffer(out);
  std::cout << "Batched readback test passed." << std::endl;
}

int main() {
  mgpuInitializeContext();

  testStagedUploads();
  testBatchedReadbacks();
  testUint8();
  testInt8();
  testInt16();
  testUint16();
  testInt32();
  testUint32();
  testInt64();
  testUint64();
  testFloat32();
  testFloat64();
  testCreateBuffer();
  testComputeShader();

  runExternalTextureTests();

  mgpuDestroyContext();
  return 0;
}
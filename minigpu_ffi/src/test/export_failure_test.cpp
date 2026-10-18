// Native gate for the failure contracts of the C exports.
//
// A failure an export can report goes through its existing return value or
// completion, never as a C++ exception through extern "C": a negative size or a
// destroyed/unknown context handle creates nothing, scope calls on such a
// handle return 0, a refused context init still calls back, an out-of-range
// typed write is logged and leaves the buffer as it was, and a staged range
// whose end would wrap past SIZE_MAX is refused. Also gated: the default log
// level is WARN, a log port is cleared only by its owner, and an unaligned raw
// write/read round-trips.
#include "../include/minigpu.h"

#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <thread>

namespace {

// BufferDataType values (buffer.h).
constexpr int kFloat32 = 0;
constexpr int kUInt32 = 2;
constexpr uint32_t kStorage = 128;

std::atomic<int> infoLines{0};
std::atomic<int> errorLines{0};
std::atomic<bool> initCalledBack{false};

void countLog(int level, const char *message) {
  if (level == 1) infoLines.fetch_add(1, std::memory_order_relaxed);
  if (level >= 3) errorLines.fetch_add(1, std::memory_order_relaxed);
  mgpuFreeLogMessage(message);
}

void onInit() { initCalledBack.store(true, std::memory_order_release); }

void require(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(1);
  }
}

void waitForFlag(std::atomic<bool> &flag, const std::string &timeoutMessage) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!flag.load(std::memory_order_acquire)) {
    require(std::chrono::steady_clock::now() < deadline, timeoutMessage);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void testLogPort() {
  // No Dart API is initialized in this process, so a registered port posts
  // nothing; only the registration is exercised.
  mgpuSetLogPort(42);
  require(mgpuClearLogPortIf(41) == 0,
          "another port must not clear the registration");
  require(mgpuClearLogPortIf(42) == 1, "the registered port must clear");
  require(mgpuClearLogPortIf(42) == 0, "a cleared port must not clear again");
  require(mgpuClearLogPortIf(0) == 0, "port 0 never reports a clear");
}

void testDefaultLogLevel() {
  // The callback was installed before init, and nothing has changed the
  // level since: no INFO line may have been delivered.
  require(infoLines.load() == 0,
          "INFO lines must be filtered at the default level");
  MGPUBuffer *buffer = mgpuCreateBuffer(16, kFloat32);
  require(buffer != nullptr, "float buffer allocation failed");
  const float values[4] = {1, 2, 3, 4};
  mgpuWriteFloat(buffer, values, sizeof(values));
  require(infoLines.load() == 0,
          "a typed write must not log INFO at the default level");
  mgpuSetLogLevel(1);
  mgpuWriteFloat(buffer, values, sizeof(values));
  require(infoLines.load() > 0, "INFO lines must be delivered once enabled");
  mgpuSetLogLevel(2);
  mgpuDestroyBuffer(buffer);
}

void requireRefused(MGPUContextHandle *handle, const std::string &what) {
  require(mgpuContextCreateBuffer(handle, 16, kUInt32) == nullptr,
          what + " handle: CreateBuffer must return NULL");
  require(mgpuContextCreateBufferWithUsage(handle, 16, kStorage) == nullptr,
          what + " handle: CreateBufferWithUsage must return NULL");
  require(mgpuContextCreateComputeShader(handle) == nullptr,
          what + " handle: CreateComputeShader must return NULL");
  char name[8] = {'x', 0};
  require(mgpuContextGetAdapterName(handle, name, sizeof(name)) == 0 &&
              name[0] == 'x',
          what + " handle: GetAdapterName must return 0 and write nothing");
  require(mgpuContextBeginUploads(handle) == 0,
          what + " handle: BeginUploads must return 0");
  require(mgpuContextEndUploads(handle) == 0,
          what + " handle: EndUploads must return 0");
  require(mgpuContextBeginReadbacks(handle) == 0,
          what + " handle: BeginReadbacks must return 0");
  require(mgpuContextEndReadbacks(handle) == 0,
          what + " handle: EndReadbacks must return 0");
  initCalledBack.store(false, std::memory_order_release);
  mgpuContextInitializeAsync(handle, onInit);
  waitForFlag(initCalledBack,
              what + " handle: a refused init must still call back");
}

void testCreationRefusals() {
  require(mgpuCreateBuffer(-4, kUInt32) == nullptr,
          "a negative size must create nothing");
  uint64_t maxBufferSize = 0, maxBinding = 0, generation = 0;
  require(mgpuGetCurrentDeviceBufferLimits(&maxBufferSize, &maxBinding,
                                           &generation) == 1,
          "the default device's limits are unavailable");
  if (maxBufferSize + 4 <= static_cast<uint64_t>(INT_MAX)) {
    require(mgpuCreateBuffer(static_cast<int>(maxBufferSize + 4), kUInt32) ==
                nullptr,
            "a size over maxBufferSize must create nothing");
  } else {
    std::cout << "skip: maxBufferSize " << maxBufferSize
              << " is past the int size argument" << std::endl;
  }
  int local = 0;
  requireRefused(reinterpret_cast<MGPUContextHandle *>(&local), "unknown");
}

void testOutOfRangeWrite() {
  MGPUBuffer *buffer = mgpuCreateBuffer(16, kUInt32);
  require(buffer != nullptr, "uint32 buffer allocation failed");
  const uint32_t first[4] = {1, 2, 3, 4};
  mgpuWriteUint32(buffer, first, sizeof(first));
  const uint32_t tooMany[8] = {9, 9, 9, 9, 9, 9, 9, 9};
  const int errorsBefore = errorLines.load();
  mgpuWriteUint32(buffer, tooMany, sizeof(tooMany));
  require(errorLines.load() > errorsBefore,
          "an out-of-range write must be logged");
  uint32_t out[4] = {};
  mgpuReadSyncUint32(buffer, out, 4, 0);
  require(std::memcmp(out, first, sizeof(first)) == 0,
          "an out-of-range write must leave the buffer unchanged");

  // Outside a scope an unaligned stage call runs inline; one that cannot
  // happen reports -1 instead of leaving the destination as it was.
  uint8_t bytes[64] = {};
  require(mgpuStageWrite(buffer, 3, bytes, sizeof(bytes)) == -1,
          "an inline stage write past the end must return -1");
  require(mgpuStageRead(buffer, 3, bytes, sizeof(bytes)) == -1,
          "an inline stage read past the end must return -1");
  mgpuDestroyBuffer(buffer);
}

void testStageRangeWrap() {
  MGPUBuffer *buffer = mgpuCreateBuffer(64, kUInt32);
  require(buffer != nullptr, "uint32 buffer allocation failed");
  // 4-byte aligned, so the range reaches the staged path; offset + size wraps
  // to 4, which a sum-form bound check would accept.
  const size_t wrapOffset = (std::numeric_limits<size_t>::max)() - 3;
  uint32_t words[2] = {5, 6};
  require(mgpuBeginUploads() == 1, "upload scope did not open");
  require(mgpuStageWrite(buffer, wrapOffset, words, sizeof(words)) == -1,
          "a staged write whose end wraps must be refused");
  require(mgpuEndUploads() == 0, "a refused write must stage nothing");
  require(mgpuBeginReadbacks() == 1, "readback scope did not open");
  require(mgpuStageRead(buffer, wrapOffset, words, sizeof(words)) == -1,
          "a staged read whose end wraps must be refused");
  require(mgpuEndReadbacks() == 0, "a refused read must stage nothing");
  mgpuDestroyBuffer(buffer);
}

void testUnalignedRaw() {
  // The legacy creator's usage includes CopySrc, which an unaligned write
  // needs to read its edge words back.
  MGPUBuffer *buffer = mgpuCreateBuffer(16, kUInt32);
  require(buffer != nullptr, "uint32 buffer allocation failed");
  const uint8_t zeros[16] = {};
  mgpuWriteBufferAt(buffer, zeros, sizeof(zeros), 0);
  const uint8_t bytes[3] = {0xAA, 0xBB, 0xCC};
  mgpuWriteBufferAt(buffer, bytes, sizeof(bytes), 5);
  uint8_t out[7] = {};
  require(mgpuStageRead(buffer, 3, out, sizeof(out)) == 0,
          "an unaligned read outside a scope must run inline");
  const uint8_t expected[7] = {0, 0, 0xAA, 0xBB, 0xCC, 0, 0};
  require(std::memcmp(out, expected, sizeof(out)) == 0,
          "an unaligned raw write/read must round-trip");
  mgpuDestroyBuffer(buffer);
}

void testDestroyedHandle() {
  MGPUContextHandle *context = mgpuCreateContextHandle(nullptr);
  require(context != nullptr, "context handle creation failed");
  mgpuDestroyContextHandle(context);
  // Retired at once: refused, never dereferenced. Runs last, so no later
  // context can be allocated at the same address.
  requireRefused(context, "destroyed");
}

} // namespace

int main() {
  mgpuSetLogCallback(countLog);
  mgpuInitializeContext();
  testLogPort();
  testDefaultLogLevel();
  testCreationRefusals();
  testOutOfRangeWrite();
  testStageRangeWrap();
  testUnalignedRaw();
  mgpuDrainWorkQueue();
  testDestroyedHandle();
  mgpuDestroyContext();
  mgpuSetLogCallback(nullptr);
  std::cout << "Export failure contract test passed." << std::endl;
  return 0;
}

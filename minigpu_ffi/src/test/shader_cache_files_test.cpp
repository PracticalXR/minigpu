#include "../include/minigpu.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

// A configured cache directory can be the host's own folder (Android has no
// default, so the host hands one over): scan, clear and eviction must count
// and delete only the files the cache writes. A hit must refresh an entry's
// time, so eviction takes the least recently USED entries first.

namespace fs = std::filesystem;

namespace {

constexpr int kF32 = 1; // mgpuCreateBuffer data type code
constexpr int kCount = 64;
constexpr int kBytes = kCount * static_cast<int>(sizeof(float));

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

void require(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(1);
  }
}

void writeFile(const fs::path &path, size_t bytes) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  const std::string data(bytes, 'x');
  file.write(data.data(), static_cast<std::streamsize>(data.size()));
  require(static_cast<bool>(file), "could not seed a test file");
}

MGPUShaderCacheStats stats() {
  MGPUShaderCacheStats out = {};
  mgpuGetShaderCacheStats(&out);
  return out;
}

// Every regular file directly in `dir` that is not one of the host's.
std::vector<fs::path> cacheFiles(const fs::path &dir,
                                 const std::vector<fs::path> &foreign) {
  std::vector<fs::path> found;
  std::error_code ec;
  for (fs::directory_iterator it(dir, ec), end; !ec && it != end;
       it.increment(ec)) {
    if (!it->is_regular_file()) continue;
    bool host = false;
    for (const auto &p : foreign) host = host || p == it->path();
    if (!host) found.push_back(it->path());
  }
  return found;
}

// Compiles and runs `kernel`, so the device creates (or loads) its pipeline.
void runKernel(const char *kernel, float mul) {
  float input[kCount];
  for (int i = 0; i < kCount; ++i) input[i] = static_cast<float>(i);
  MGPUBuffer *inp = mgpuCreateBuffer(kBytes, kF32);
  MGPUBuffer *out = mgpuCreateBuffer(kBytes, kF32);
  MGPUComputeShader *shader = mgpuCreateComputeShader();
  require(inp && out && shader, "buffer or shader creation failed");
  mgpuWriteFloat(inp, input, sizeof(input));
  mgpuLoadKernel(shader, kernel);
  mgpuSetBuffer(shader, 0, inp);
  mgpuSetBuffer(shader, 1, out);
  mgpuDispatch(shader, 1, 1, 1);
  mgpuDrainWorkQueue();
  float output[kCount] = {};
  mgpuReadSync(out, output, sizeof(output), 0);
  require(output[5] == 5.0f * mul + 1.0f, "kernel produced a wrong result");
  mgpuDestroyComputeShader(shader);
  mgpuDestroyBuffer(out);
  mgpuDestroyBuffer(inp);
  mgpuDrainWorkQueue();
}

} // namespace

int main() {
  // The environment beats a configured directory; this test configures one.
#ifdef _WIN32
  _putenv_s("MGPU_SHADER_CACHE", "");
  _putenv_s("MGPU_SHADER_CACHE_DIR", "");
#else
  unsetenv("MGPU_SHADER_CACHE");
  unsetenv("MGPU_SHADER_CACHE_DIR");
#endif

  std::error_code ec;
  const fs::path dir = fs::absolute("shader_cache_files_test.dir");
  fs::remove_all(dir, ec);
  fs::create_directories(dir / "nested", ec);
  require(fs::is_directory(dir / "nested"), "could not create the test dir");

  // What the cache writes: an entry and a store's leftover temp file.
  const fs::path entry = dir / (std::string(64, 'a') + ".blob");
  const fs::path leftover = dir / (std::string(64, 'b') + ".blob.tmp-1-1");
  writeFile(entry, 100);
  writeFile(leftover, 50);
  // The host's files, near-misses of those names included.
  const std::vector<fs::path> foreign = {
      dir / "settings.json",
      dir / (std::string(64, 'F') + ".blob"),
      dir / (std::string(64, 'c') + ".blob.bak"),
      dir / (std::string(63, 'd') + ".blob"),
      dir / "nested" / (std::string(64, 'e') + ".blob"),
  };
  for (const auto &p : foreign) writeFile(p, 7);

  mgpuShaderCacheSetEnabled(1);
  mgpuShaderCacheSetCapBytes(0); // no eviction until the eviction check
  mgpuShaderCacheSetDirectory(dir.string().c_str());
  char resolved[1024] = {};
  require(mgpuGetShaderCacheDirectory(resolved, sizeof(resolved)) > 0,
          "the test directory did not resolve");

  const MGPUShaderCacheStats seeded = stats();
  require(seeded.entryCount == 2 && seeded.bytesOnDisk == 150,
          "the scan counted files the cache did not write");
  require(mgpuShaderCacheClear() == 2,
          "clear did not remove exactly the cache's two files");
  require(!fs::exists(entry) && !fs::exists(leftover), "clear kept a cache file");
  for (const auto &p : foreign) require(fs::exists(p), "clear deleted a host file");

  // Month-old host files would be first in line for an mtime-ordered eviction.
  const auto now = fs::file_time_type::clock::now();
  for (const auto &p : foreign)
    fs::last_write_time(p, now - std::chrono::hours(24 * 30), ec);

  mgpuInitializeContext();
  runKernel(kDouble, 2.0f);
  mgpuDestroyContext();
  const MGPUShaderCacheStats cold = stats();
  const std::vector<fs::path> blobs = cacheFiles(dir, foreign);
  if (cold.stores == 0 || blobs.empty()) {
    std::cout << "shader_cache_files_test: the backend stored no blobs; hit "
                 "and eviction checks skipped" << std::endl;
    fs::remove_all(dir, ec);
    return 0;
  }

  // A hit on a new device refreshes the entry's time.
  for (const auto &p : blobs)
    fs::last_write_time(p, now - std::chrono::hours(48), ec);
  mgpuInitializeContext();
  runKernel(kDouble, 2.0f);
  const MGPUShaderCacheStats warm = stats();
  require(warm.hits > cold.hits, "the second device did not hit the cache");
  bool refreshed = false;
  for (const auto &p : blobs) {
    const auto stamp = fs::last_write_time(p, ec);
    refreshed = refreshed || (!ec && stamp > now - std::chrono::hours(1));
  }
  require(refreshed, "a cache hit did not refresh the entry's time");

  // Over the cap, eviction deletes cache files only, however old the rest.
  mgpuShaderCacheSetCapBytes(1);
  runKernel(kTriple, 3.0f); // new kernel: miss, store, evict
  mgpuDestroyContext();
  const MGPUShaderCacheStats evicted = stats();
  require(evicted.stores > warm.stores, "the new kernel was not stored");
  require(evicted.evictions > warm.evictions, "nothing was evicted over the cap");
  for (const auto &p : foreign)
    require(fs::exists(p), "eviction deleted a host file");
  require(cacheFiles(dir, foreign).empty(),
          "eviction kept a cache file over a 1-byte cap");

  fs::remove_all(dir, ec);
  std::cout << "shader_cache_files_test passed (stores=" << evicted.stores
            << " hits=" << evicted.hits << " evictions=" << evicted.evictions
            << ")" << std::endl;
  return 0;
}

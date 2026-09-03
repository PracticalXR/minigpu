// Persistent shader cache — storage module.
//
// Turns WGSL->backend shader compilation into a once-per-machine cost instead
// of a once-per-process one, by giving Dawn's BlobCache somewhere durable to
// put its compiled blobs. See shader_cache.h for the interface contract and
// buffer.cpp for the device-descriptor wiring.
//
// DESIGN RULES, in the order they matter:
//
//  1. A failure is ALWAYS a cache miss, never a failed device. Directory
//     unwritable, disk full, antivirus lock, corrupt entry, another process
//     mid-write — every one of them degrades to "compile it" and continues.
//     A cache that can break startup is worse than no cache, because the
//     failure lands on users who were previously fine. Nothing in this file
//     may throw out of a Dawn callback.
//
//  2. The key is a correctness boundary, not an optimisation. A stale blob is
//     not a slow frame; it is a wrong pipeline. Every entry stores its FULL
//     key and load() compares it — see kMagic below for why the filename hash
//     alone is not enough.
//
//  3. Dawn's callbacks are synchronous and can arrive on Dawn-internal
//     threads, so all state sits behind one mutex and nothing here calls into
//     Dart. (A Dart provider is impossible by construction: entering an
//     isolate from a foreign thread requires NativeCallable.listener, which is
//     asynchronous and cannot return a blob to a blocked native caller.)

#include "../include/shader_cache.h"

#include "../include/log.h"
#include "../include/minigpu.h"
#include "../include/mutex.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#ifndef __EMSCRIPTEN__

#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <system_error>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

// ── On-disk format ──────────────────────────────────────────────────────────
//
//   magic "MGPUSC01"   8 bytes — ALSO the format version (R2.3): bumping it
//                                makes every old entry unreachable rather
//                                than merely unused, because the magic check
//                                fails before anything else is trusted.
//   u32   keyLen       little-endian
//   key   keyLen bytes the FULL Dawn cache key
//   value rest         Dawn's blob, byte for byte
//
// The full key is stored because the filename is only a hash of it. Dawn's own
// hash validation binds hash->value, NOT value->key: on a filename collision
// it would happily accept a blob that is internally consistent but belongs to
// a different pipeline. Comparing the stored key on load is the only thing
// standing between a hash collision and a silently wrong pipeline.
constexpr char kMagic[8] = {'M', 'G', 'P', 'U', 'S', 'C', '0', '1'};
constexpr size_t kHeaderSize = sizeof(kMagic) + sizeof(uint32_t);

// Cache directory namespace. Kept in step with kMagic — a format change that
// needs old entries gone can bump either.
constexpr const char *kCacheNamespace = "minigpu";
constexpr const char *kCacheVersionDir = "v1";

constexpr uint64_t kDefaultCapBytes = 256ull * 1024 * 1024;

// ── SHA-256 ─────────────────────────────────────────────────────────────────
// Used only to derive a filename from a key. A collision costs a recompile
// (the stored-key compare rejects it), never a wrong result — but at 256 bits
// the question does not come up.
struct Sha256 {
  uint32_t state[8];
  uint64_t bitLen;
  uint8_t buf[64];
  size_t bufLen;

  static uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
  }

  void init() {
    state[0] = 0x6a09e667; state[1] = 0xbb67ae85;
    state[2] = 0x3c6ef372; state[3] = 0xa54ff53a;
    state[4] = 0x510e527f; state[5] = 0x9b05688c;
    state[6] = 0x1f83d9ab; state[7] = 0x5be0cd19;
    bitLen = 0;
    bufLen = 0;
  }

  void transform(const uint8_t *chunk) {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (uint32_t)chunk[i * 4] << 24 | (uint32_t)chunk[i * 4 + 1] << 16 |
             (uint32_t)chunk[i * 4 + 2] << 8 | (uint32_t)chunk[i * 4 + 3];
    }
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 =
          rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 =
          rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const uint32_t ch = (e & f) ^ ((~e) & g);
      const uint32_t t1 = h + S1 + ch + k[i] + w[i];
      const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = S0 + maj;
      h = g; g = f; f = e; e = d + t1;
      d = c; c = b; b = a; a = t1 + t2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
  }

  void update(const uint8_t *data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
      buf[bufLen++] = data[i];
      if (bufLen == 64) {
        transform(buf);
        bitLen += 512;
        bufLen = 0;
      }
    }
  }

  void final(uint8_t out[32]) {
    size_t i = bufLen;
    if (bufLen < 56) {
      buf[i++] = 0x80;
      while (i < 56) buf[i++] = 0x00;
    } else {
      buf[i++] = 0x80;
      while (i < 64) buf[i++] = 0x00;
      transform(buf);
      std::memset(buf, 0, 56);
    }
    bitLen += (uint64_t)bufLen * 8;
    for (int j = 0; j < 8; ++j) buf[63 - j] = (uint8_t)(bitLen >> (j * 8));
    transform(buf);
    for (int j = 0; j < 4; ++j) {
      for (int s = 0; s < 8; ++s) {
        out[j + s * 4] = (uint8_t)((state[s] >> (24 - j * 8)) & 0xff);
      }
    }
  }
};

std::string hexDigest(const void *data, size_t len) {
  Sha256 h;
  h.init();
  h.update(static_cast<const uint8_t *>(data), len);
  uint8_t d[32];
  h.final(d);
  static const char *hex = "0123456789abcdef";
  std::string s;
  s.reserve(64);
  for (unsigned char b : d) {
    s.push_back(hex[b >> 4]);
    s.push_back(hex[b & 0x0f]);
  }
  return s;
}

// ── Path helpers ────────────────────────────────────────────────────────────
// %LOCALAPPDATA% and $HOME can contain non-ASCII; everything goes through
// std::filesystem::path so nothing is ever routed via a narrow ANSI API.
fs::path pathFromUtf8(const char *s) {
  if (!s) return {};
#if defined(__cpp_char8_t)
  return fs::path(reinterpret_cast<const char8_t *>(s));
#else
  return fs::u8path(s);
#endif
}

std::string utf8FromPath(const fs::path &p) {
  const auto u8 = p.u8string();
  return std::string(reinterpret_cast<const char *>(u8.data()), u8.size());
}

/// The OS cache convention per platform, plus namespace and format version.
/// An OS purging this directory is a SUPPORTED event: it costs one slow
/// launch and nothing else.
fs::path defaultCacheDir() {
#ifdef _WIN32
  const wchar_t *base = _wgetenv(L"LOCALAPPDATA");
  if (!base || !*base) {
    // Older/atypical environments: derive it rather than give up.
    const wchar_t *profile = _wgetenv(L"USERPROFILE");
    if (!profile || !*profile) return {};
    return fs::path(profile) / L"AppData" / L"Local" / kCacheNamespace /
           L"shadercache" / kCacheVersionDir;
  }
  return fs::path(base) / kCacheNamespace / L"shadercache" / kCacheVersionDir;
#elif defined(__ANDROID__)
  // An app-private cache dir cannot be discovered from JNI-free C++, and
  // guessing one would mean writing somewhere the OS never reclaims. The host
  // must supply it via mgpuShaderCacheSetDirectory(); until then the cache
  // stays inert.
  return {};
#elif defined(__APPLE__)
  const char *home = std::getenv("HOME");
  if (!home || !*home) return {};
  return fs::path(home) / "Library" / "Caches" / kCacheNamespace /
         "shadercache" / kCacheVersionDir;
#else
  if (const char *xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg) {
    return fs::path(xdg) / kCacheNamespace / "shadercache" / kCacheVersionDir;
  }
  const char *home = std::getenv("HOME");
  if (!home || !*home) return {};
  return fs::path(home) / ".cache" / kCacheNamespace / "shadercache" /
         kCacheVersionDir;
#endif
}

// ── Cache state ─────────────────────────────────────────────────────────────
// One process-global singleton, intentionally never destroyed: Dawn holds the
// function pointers for the lifetime of every device it creates and may call
// them from its own threads during teardown. A destructor here would be a
// use-after-free waiting for the right shutdown ordering.
struct CacheState {
  mgpu::mutex mu;

  bool enabled = true;
  bool dirResolved = false;   // directory usable (created / writable)
  bool dirAttempted = false;  // resolution already tried this process
  fs::path dir;
  fs::path configuredDir;  // explicit override, empty = use the OS default
  uint64_t capBytes = kDefaultCapBytes;
  std::string extraKey;

  // Environment overrides, read once. These OUTRANK the programmatic setters,
  // same as MGPU_ADAPTER_NAME outranks preferDisplayAdapter: the point of an
  // env var is to change the behaviour of a binary you cannot edit, so a
  // caller that hard-codes a setting must not be able to defeat it.
  bool envApplied = false;
  bool envDisabled = false;
  fs::path envDir;

  MGPUShaderCacheLoadFn providerLoad = nullptr;
  MGPUShaderCacheStoreFn providerStore = nullptr;
  void *providerUser = nullptr;

  // Single-entry memo bridging Dawn's two load calls. Dawn sizes an entry,
  // then asks for it; serving the second call from here halves the I/O and,
  // more usefully, guarantees the two calls agree even if the file changes in
  // between. A miss just falls back to re-reading, so concurrent compiles on
  // several threads stay correct.
  std::vector<uint8_t> memoKey;
  std::vector<uint8_t> memoValue;
  bool memoValid = false;

  uint64_t hits = 0, misses = 0, stores = 0, storeFailures = 0, evictions = 0;
  uint64_t bytesOnDisk = 0, entryCount = 0;
  uint64_t loadUs = 0, storeUs = 0;
  uint64_t pipelineUs = 0;
  bool scanned = false;

  uint64_t tempCounter = 0;
};

CacheState &state() {
  static CacheState *s = new CacheState();
  return *s;
}

using Clock = std::chrono::steady_clock;

uint64_t usSince(Clock::time_point t0) {
  return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
             Clock::now() - t0)
      .count();
}

/// Reads the environment overrides once. Called with the lock held.
///
///   MGPU_SHADER_CACHE=0|off|false|no   turn caching off entirely
///   MGPU_SHADER_CACHE_DIR=<path>       store blobs here instead
///
/// This is the same escape hatch MGPU_BACKEND / MGPU_ADAPTER_NAME /
/// MGPU_WAIT_PROF provide: a way to answer "is the cache the problem?" on an
/// already-built binary, without a code change and a rebuild.
void applyEnvOnceLocked(CacheState &s) {
  if (s.envApplied) return;
  s.envApplied = true;

  if (const char *v = std::getenv("MGPU_SHADER_CACHE"); v && v[0]) {
    std::string val(v);
    for (char &c : val) c = (char)std::tolower((unsigned char)c);
    if (val == "0" || val == "off" || val == "false" || val == "no") {
      s.envDisabled = true;
      MGPU_LOG(mgpu::LOG_INFO,
               "[mgpu shadercache] disabled by MGPU_SHADER_CACHE=%s", v);
    }
  }

  if (const char *d = std::getenv("MGPU_SHADER_CACHE_DIR"); d && d[0]) {
    s.envDir = pathFromUtf8(d);
    MGPU_LOG(mgpu::LOG_INFO,
             "[mgpu shadercache] directory overridden by "
             "MGPU_SHADER_CACHE_DIR=%s", d);
  }
}

/// Caching is on only if nothing — code or environment — turned it off.
bool cacheEnabledLocked(CacheState &s) {
  applyEnvOnceLocked(s);
  return s.enabled && !s.envDisabled;
}

/// Resolves and creates the cache directory once per process. Called with the
/// lock held. Any failure leaves dirResolved false, which makes every
/// subsequent operation an immediate miss.
void ensureDirLocked(CacheState &s) {
  applyEnvOnceLocked(s);
  if (s.dirAttempted) return;
  s.dirAttempted = true;

  // MGPU_SHADER_CACHE_DIR > configured directory > per-platform default.
  fs::path target = !s.envDir.empty()
      ? s.envDir
      : (s.configuredDir.empty() ? defaultCacheDir() : s.configuredDir);
  if (target.empty()) {
    MGPU_LOG(mgpu::LOG_WARN,
             "[mgpu shadercache] no cache directory available on this "
             "platform; shader caching is off (set one explicitly to enable)");
    return;
  }

  std::error_code ec;
  fs::create_directories(target, ec);
  // create_directories reports "already exists" as a non-error via ec only on
  // some implementations, so confirm by asking rather than trusting ec.
  if (!fs::is_directory(target, ec)) {
    MGPU_LOG(mgpu::LOG_WARN,
             "[mgpu shadercache] cannot use '%s' (%s); shaders will compile "
             "every launch",
             utf8FromPath(target).c_str(), ec ? ec.message().c_str() : "not a directory");
    return;
  }

  s.dir = std::move(target);
  s.dirResolved = true;
}

/// Seeds bytesOnDisk / entryCount from the directory once, so the size cap and
/// the stats mean something on a warm start. Called with the lock held.
void scanLocked(CacheState &s) {
  if (s.scanned || !s.dirResolved) return;
  s.scanned = true;

  std::error_code ec;
  uint64_t bytes = 0, count = 0;
  for (fs::directory_iterator it(s.dir, ec), end; !ec && it != end;
       it.increment(ec)) {
    std::error_code fe;
    if (!it->is_regular_file(fe) || fe) continue;
    const auto sz = it->file_size(fe);
    if (fe) continue;
    bytes += sz;
    ++count;
  }
  s.bytesOnDisk = bytes;
  s.entryCount = count;
}

/// LRU eviction by last-write time. Runs only when the running total exceeds
/// the cap, and evicts down to 90% of it so a store near the boundary does not
/// rescan the directory every time.
///
/// This runs on the compile path, which on a miss is already seconds of FXC —
/// a directory scan is noise there. Deleting an entry another process is
/// reading is safe: readers open with full sharing, so the unlink is deferred.
void evictLocked(CacheState &s) {
  if (!s.dirResolved || s.capBytes == 0) return;
  if (s.bytesOnDisk <= s.capBytes) return;

  struct Entry {
    fs::path path;
    uint64_t size;
    fs::file_time_type mtime;
  };
  std::vector<Entry> entries;

  std::error_code ec;
  for (fs::directory_iterator it(s.dir, ec), end; !ec && it != end;
       it.increment(ec)) {
    std::error_code fe;
    if (!it->is_regular_file(fe) || fe) continue;
    const auto sz = it->file_size(fe);
    if (fe) continue;
    const auto mt = it->last_write_time(fe);
    if (fe) continue;
    entries.push_back({it->path(), sz, mt});
  }

  std::sort(entries.begin(), entries.end(),
            [](const Entry &a, const Entry &b) { return a.mtime < b.mtime; });

  const uint64_t targetBytes = s.capBytes - s.capBytes / 10;
  uint64_t live = 0;
  for (const auto &e : entries) live += e.size;

  for (const auto &e : entries) {
    if (live <= targetBytes) break;
    std::error_code de;
    if (fs::remove(e.path, de) && !de) {
      live -= e.size;
      ++s.evictions;
      if (s.entryCount > 0) --s.entryCount;
    } else {
      // Another process holds it, or it is already gone. Either way it is not
      // ours to insist on — stop counting it as reclaimable and move on.
      live -= e.size;
    }
  }
  s.bytesOnDisk = live;
}

fs::path entryPathLocked(const CacheState &s, const void *key, size_t keyLen) {
  return s.dir / (hexDigest(key, keyLen) + ".blob");
}

/// Reads an entry and returns its payload, or false for any miss reason:
/// absent, unreadable, wrong magic, truncated, or — the one that matters —
/// a stored key that is not the requested key.
bool readEntry(const fs::path &p, const void *key, size_t keyLen,
               std::vector<uint8_t> &outValue) {
  std::error_code ec;
  const auto szRaw = fs::file_size(p, ec);
  if (ec) return false;
  const size_t sz = (size_t)szRaw;
  if (sz <= kHeaderSize + keyLen) return false;

  std::vector<uint8_t> raw(sz);

#ifdef _WIN32
  // FILE_SHARE_DELETE matters: eviction from another process must never fault
  // a read that is already in flight.
  HANDLE h = CreateFileW(p.c_str(), GENERIC_READ,
                         FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                         nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE) return false;
  size_t got = 0;
  while (got < sz) {
    DWORD chunk = (DWORD)std::min<size_t>(sz - got, 1u << 20);
    DWORD read = 0;
    if (!ReadFile(h, raw.data() + got, chunk, &read, nullptr) || read == 0) {
      CloseHandle(h);
      return false;
    }
    got += read;
  }
  CloseHandle(h);
#else
  FILE *f = std::fopen(p.c_str(), "rb");
  if (!f) return false;
  const size_t got = std::fread(raw.data(), 1, sz, f);
  std::fclose(f);
  if (got != sz) return false;
#endif

  if (std::memcmp(raw.data(), kMagic, sizeof(kMagic)) != 0) return false;

  uint32_t storedKeyLen = 0;
  std::memcpy(&storedKeyLen, raw.data() + sizeof(kMagic), sizeof(uint32_t));
  if (storedKeyLen != keyLen) return false;
  if (kHeaderSize + (size_t)storedKeyLen >= sz) return false;
  // THE check (see kMagic): the filename is only a hash, so without comparing
  // the stored key a collision would hand Dawn a blob for a different
  // pipeline — one that passes Dawn's own hash validation, because that binds
  // hash to value and says nothing about which key the value belongs to.
  if (std::memcmp(raw.data() + kHeaderSize, key, keyLen) != 0) return false;

  const size_t valueOff = kHeaderSize + storedKeyLen;
  outValue.assign(raw.begin() + (long long)valueOff, raw.end());
  return !outValue.empty();
}

/// Writes an entry atomically: a temp file in the SAME directory, flushed to
/// disk, then renamed over the target. A reader must never observe a
/// half-written blob; a torn write that survives anyway is caught by the magic
/// / key checks on load and by Dawn's hash validation, so it degrades to a
/// miss rather than corruption.
bool writeEntryAtomic(const fs::path &target, const fs::path &tmp,
                      const void *key, size_t keyLen, const void *value,
                      size_t valueLen) {
  std::vector<uint8_t> blob;
  blob.reserve(kHeaderSize + keyLen + valueLen);
  blob.insert(blob.end(), kMagic, kMagic + sizeof(kMagic));
  const uint32_t kl = (uint32_t)keyLen;
  const uint8_t *klp = reinterpret_cast<const uint8_t *>(&kl);
  blob.insert(blob.end(), klp, klp + sizeof(uint32_t));
  const uint8_t *kp = static_cast<const uint8_t *>(key);
  blob.insert(blob.end(), kp, kp + keyLen);
  const uint8_t *vp = static_cast<const uint8_t *>(value);
  blob.insert(blob.end(), vp, vp + valueLen);

#ifdef _WIN32
  HANDLE h = CreateFileW(tmp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                         FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE) return false;
  size_t off = 0;
  while (off < blob.size()) {
    DWORD chunk = (DWORD)std::min<size_t>(blob.size() - off, 1u << 20);
    DWORD wrote = 0;
    if (!WriteFile(h, blob.data() + off, chunk, &wrote, nullptr) || wrote == 0) {
      CloseHandle(h);
      std::error_code ec;
      fs::remove(tmp, ec);
      return false;
    }
    off += wrote;
  }
  FlushFileBuffers(h);
  CloseHandle(h);
#else
  const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) return false;
  size_t off = 0;
  while (off < blob.size()) {
    const ssize_t wrote = ::write(fd, blob.data() + off, blob.size() - off);
    if (wrote <= 0) {
      ::close(fd);
      std::error_code ec;
      fs::remove(tmp, ec);
      return false;
    }
    off += (size_t)wrote;
  }
  ::fsync(fd);
  ::close(fd);
#endif

  std::error_code ec;
  fs::rename(tmp, target, ec);
  if (ec) {
    // Losing the rename race with another process is fine — the winner wrote
    // the same content under the same key.
    std::error_code re;
    fs::remove(tmp, re);
    return false;
  }
  return true;
}

} // namespace

// ── Internal interface (shader_cache.h) ─────────────────────────────────────

namespace mgpu {

bool shaderCacheIsActive() {
  CacheState &s = state();
  mgpu::lock_guard<mgpu::mutex> lock(s.mu);
  if (!cacheEnabledLocked(s)) return false;
  if (s.providerLoad || s.providerStore) return true;
  ensureDirLocked(s);
  return s.dirResolved;
}

std::string shaderCacheIsolationKey(uint32_t vendorId, uint32_t deviceId,
                                    const std::string &adapterDesc) {
  CacheState &s = state();
  std::string extra;
  {
    mgpu::lock_guard<mgpu::mutex> lock(s.mu);
    extra = s.extraKey;
  }
  char head[128];
  std::snprintf(head, sizeof(head), "mgpu-sc1|v=%08X|d=%08X|", vendorId,
                deviceId);
  return std::string(head) + adapterDesc + "|" + extra;
}

std::string shaderCacheSummaryLine() {
  CacheState &s = state();
  mgpu::lock_guard<mgpu::mutex> lock(s.mu);
  if (!cacheEnabledLocked(s)) {
    return s.envDisabled ? "[mgpu shadercache] disabled by MGPU_SHADER_CACHE"
                         : "[mgpu shadercache] disabled";
  }
  const bool byo = s.providerLoad || s.providerStore;
  if (!byo && !s.dirResolved) {
    return "[mgpu shadercache] inactive (no usable directory) — shaders "
           "compile every launch";
  }
  scanLocked(s);
  char buf[512];
  std::snprintf(buf, sizeof(buf),
                "[mgpu shadercache] %s entries=%llu bytes=%llu cap=%lluMB "
                "hits=%llu misses=%llu stores=%llu evictions=%llu",
                byo ? "provider=custom" : utf8FromPath(s.dir).c_str(),
                (unsigned long long)s.entryCount,
                (unsigned long long)s.bytesOnDisk,
                (unsigned long long)(s.capBytes / (1024 * 1024)),
                (unsigned long long)s.hits, (unsigned long long)s.misses,
                (unsigned long long)s.stores,
                (unsigned long long)s.evictions);
  return buf;
}

void shaderCacheNotePipelineMs(double ms) {
  if (ms <= 0.0) return;
  CacheState &s = state();
  mgpu::lock_guard<mgpu::mutex> lock(s.mu);
  s.pipelineUs += (uint64_t)(ms * 1000.0);
}

} // namespace mgpu

// ── Dawn trampolines ────────────────────────────────────────────────────────

extern "C" size_t mgpu_shader_cache_dawn_load(const void *key, size_t keySize,
                                              void *value, size_t valueSize,
                                              void *userdata) {
  (void)userdata;
  if (!key || keySize == 0) return 0;

  CacheState &s = state();
  const auto t0 = Clock::now();
  mgpu::lock_guard<mgpu::mutex> lock(s.mu);
  if (!cacheEnabledLocked(s)) return 0;

  // Bring-your-own provider: forwarded verbatim, including the two-phase
  // protocol and the 0-means-miss convention.
  if (s.providerLoad) {
    const size_t n = s.providerLoad(key, keySize, value, valueSize,
                                    s.providerUser);
    if (value == nullptr) {
      if (n > 0) ++s.hits; else ++s.misses;
    }
    s.loadUs += usSince(t0);
    return n;
  }

  ensureDirLocked(s);
  if (!s.dirResolved) {
    if (value == nullptr) ++s.misses;
    return 0;
  }

  // Phase 2 fast path: Dawn is asking for the entry it just sized.
  if (value != nullptr && s.memoValid && s.memoKey.size() == keySize &&
      std::memcmp(s.memoKey.data(), key, keySize) == 0) {
    const size_t n = s.memoValue.size();
    if (n > valueSize) {
      // Cannot fit: return a size that differs from the sizing call so Dawn
      // treats it as a miss instead of reading past the buffer.
      s.memoValid = false;
      s.loadUs += usSince(t0);
      return 0;
    }
    std::memcpy(value, s.memoValue.data(), n);
    s.memoValid = false;
    s.loadUs += usSince(t0);
    return n;
  }

  std::vector<uint8_t> payload;
  bool ok = false;
  try {
    ok = readEntry(entryPathLocked(s, key, keySize), key, keySize, payload);
  } catch (...) {
    ok = false;  // A cache can never be the reason a device fails to come up.
  }

  if (!ok) {
    if (value == nullptr) {
      ++s.misses;
      MGPU_LOG(mgpu::LOG_DEBUG, "[mgpu shadercache] miss (%zu-byte key)",
               keySize);
    }
    s.loadUs += usSince(t0);
    return 0;
  }

  if (value == nullptr) {
    // Sizing call: remember the payload so phase 2 is served from memory and
    // cannot disagree with what we just reported.
    ++s.hits;
    s.memoKey.assign((const uint8_t *)key, (const uint8_t *)key + keySize);
    s.memoValue = payload;
    s.memoValid = true;
    MGPU_LOG(mgpu::LOG_DEBUG, "[mgpu shadercache] hit (%zu bytes)",
             payload.size());
    s.loadUs += usSince(t0);
    return payload.size();
  }

  if (payload.size() > valueSize) {
    s.loadUs += usSince(t0);
    return 0;
  }
  std::memcpy(value, payload.data(), payload.size());
  s.loadUs += usSince(t0);
  return payload.size();
}

extern "C" void mgpu_shader_cache_dawn_store(const void *key, size_t keySize,
                                             const void *value,
                                             size_t valueSize,
                                             void *userdata) {
  (void)userdata;
  if (!key || keySize == 0 || !value || valueSize == 0) return;

  CacheState &s = state();
  const auto t0 = Clock::now();
  mgpu::lock_guard<mgpu::mutex> lock(s.mu);
  if (!cacheEnabledLocked(s)) return;

  if (s.providerStore) {
    s.providerStore(key, keySize, value, valueSize, s.providerUser);
    ++s.stores;
    s.storeUs += usSince(t0);
    return;
  }

  ensureDirLocked(s);
  if (!s.dirResolved) return;

  try {
    scanLocked(s);
    const fs::path target = entryPathLocked(s, key, keySize);
    // Process id + counter, so two processes writing the same key at the same
    // moment cannot collide on the temp name. The base name is a hex digest
    // plus ".blob", i.e. pure ASCII — .string() cannot be lossy here.
    char suffix[64];
    std::snprintf(suffix, sizeof(suffix), ".tmp-%llu-%llu",
#ifdef _WIN32
                  (unsigned long long)GetCurrentProcessId(),
#else
                  (unsigned long long)::getpid(),
#endif
                  (unsigned long long)++s.tempCounter);
    const fs::path tmp = target.parent_path() /
                         (target.filename().string() + suffix);

    // Overwriting an existing entry replaces its bytes rather than adding to
    // them; a stat failure here only skews the size accounting, never
    // correctness, so it is deliberately not treated as a store failure.
    uint64_t oldSize = 0;
    bool replacing = false;
    {
      std::error_code ec;
      if (fs::exists(target, ec) && !ec) {
        std::error_code se;
        const auto sz = fs::file_size(target, se);
        if (!se) {
          oldSize = (uint64_t)sz;
          replacing = true;
        }
      }
    }

    if (writeEntryAtomic(target, tmp, key, keySize, value, valueSize)) {
      ++s.stores;
      const uint64_t newSize =
          (uint64_t)(kHeaderSize + keySize + valueSize);
      s.bytesOnDisk += newSize;
      if (replacing) {
        s.bytesOnDisk -= std::min<uint64_t>(s.bytesOnDisk, oldSize);
      } else {
        ++s.entryCount;
      }
      MGPU_LOG(mgpu::LOG_DEBUG, "[mgpu shadercache] stored %zu bytes",
               valueSize);
      evictLocked(s);
    } else {
      ++s.storeFailures;
      MGPU_LOG(mgpu::LOG_DEBUG,
               "[mgpu shadercache] store failed (%zu bytes) — will recompile "
               "next launch",
               valueSize);
    }
  } catch (...) {
    ++s.storeFailures;
  }
  s.storeUs += usSince(t0);
}

// ── Public C API (minigpu.h) ────────────────────────────────────────────────
//
// Every setter is PRE-INIT and PROCESS-GLOBAL, same contract as
// mgpuPreferDisplayAdapter: the value is always stored, and the return code
// says whether a live context already consumed the old one (0 = stored before
// init, 1 = a context is live so this takes effect on the next init).

// The process-global MGPU instance is DEFINED in minigpu.cpp inside an
// extern "C" block, so it must be declared with C linkage here too — the same
// declaration minigpu_external.cpp uses.
extern "C" mgpu::MGPU minigpu;

namespace {
/// 0 when no context is live, 1 when one already is. tryGetInstance() probes
/// without triggering a lazy re-initialization, which is what makes this safe
/// to call from a pre-init setter.
int contextLiveFlag() { return minigpu.tryGetInstance() ? 1 : 0; }
} // namespace

extern "C" {

int mgpuShaderCacheSetEnabled(int enabled) {
  CacheState &s = state();
  {
    mgpu::lock_guard<mgpu::mutex> lock(s.mu);
    s.enabled = enabled != 0;
  }
  return contextLiveFlag();
}

int mgpuShaderCacheSetDirectory(const char *utf8Path) {
  CacheState &s = state();
  {
    mgpu::lock_guard<mgpu::mutex> lock(s.mu);
    s.configuredDir = (utf8Path && *utf8Path) ? pathFromUtf8(utf8Path) : fs::path();
    // A new directory invalidates everything derived from the old one.
    s.dirAttempted = false;
    s.dirResolved = false;
    s.scanned = false;
    s.bytesOnDisk = 0;
    s.entryCount = 0;
    s.memoValid = false;
  }
  return contextLiveFlag();
}

int mgpuShaderCacheSetCapBytes(unsigned long long bytes) {
  CacheState &s = state();
  {
    mgpu::lock_guard<mgpu::mutex> lock(s.mu);
    s.capBytes = bytes;
  }
  return contextLiveFlag();
}

int mgpuShaderCacheSetExtraKey(const char *utf8) {
  CacheState &s = state();
  {
    mgpu::lock_guard<mgpu::mutex> lock(s.mu);
    s.extraKey = (utf8 && *utf8) ? utf8 : "";
  }
  return contextLiveFlag();
}

int mgpuShaderCacheSetProvider(MGPUShaderCacheLoadFn load,
                               MGPUShaderCacheStoreFn store, void *user) {
  CacheState &s = state();
  {
    mgpu::lock_guard<mgpu::mutex> lock(s.mu);
    s.providerLoad = load;
    s.providerStore = store;
    s.providerUser = user;
    s.memoValid = false;
  }
  return contextLiveFlag();
}

int mgpuShaderCacheClear(void) {
  CacheState &s = state();
  mgpu::lock_guard<mgpu::mutex> lock(s.mu);
  s.memoValid = false;
  ensureDirLocked(s);
  if (!s.dirResolved) return 0;

  int removed = 0;
  std::error_code ec;
  std::vector<fs::path> victims;
  for (fs::directory_iterator it(s.dir, ec), end; !ec && it != end;
       it.increment(ec)) {
    std::error_code fe;
    if (it->is_regular_file(fe) && !fe) victims.push_back(it->path());
  }
  for (const auto &p : victims) {
    std::error_code de;
    if (fs::remove(p, de) && !de) ++removed;
  }
  s.bytesOnDisk = 0;
  s.entryCount = 0;
  s.scanned = true;
  return removed;
}

void mgpuGetShaderCacheStats(MGPUShaderCacheStats *out) {
  if (!out) return;
  CacheState &s = state();
  mgpu::lock_guard<mgpu::mutex> lock(s.mu);
  scanLocked(s);
  out->hits = s.hits;
  out->misses = s.misses;
  out->stores = s.stores;
  out->storeFailures = s.storeFailures;
  out->evictions = s.evictions;
  out->bytesOnDisk = s.bytesOnDisk;
  out->entryCount = s.entryCount;
  out->loadMs = s.loadUs / 1000;
  out->storeMs = s.storeUs / 1000;
  out->pipelineCreateMs = s.pipelineUs / 1000;
  out->enabled = cacheEnabledLocked(s) ? 1u : 0u;
  out->usingDefaultProvider =
      (s.providerLoad || s.providerStore) ? 0u : 1u;
}

int mgpuGetShaderCacheDirectory(char *out, int cap) {
  CacheState &s = state();
  mgpu::lock_guard<mgpu::mutex> lock(s.mu);
  // Only resolve when caching is actually on: asking where the cache lives
  // must not CREATE a directory for a cache that will never write to it.
  if (cacheEnabledLocked(s)) ensureDirLocked(s);
  const std::string p = s.dirResolved ? utf8FromPath(s.dir) : std::string();
  if (out && cap > 0) {
    const int n = (int)p.size() < cap - 1 ? (int)p.size() : cap - 1;
    std::memcpy(out, p.c_str(), (size_t)n);
    out[n] = '\0';
  }
  return (int)p.size();
}

} // extern "C"

#else // __EMSCRIPTEN__

// Web has no Dawn native backend and no FXC — there is nothing to cache, and
// no filesystem to cache it in. The API stays present so callers need no
// conditional code; every entry point is inert.

namespace mgpu {
bool shaderCacheIsActive() { return false; }
std::string shaderCacheIsolationKey(uint32_t, uint32_t, const std::string &) {
  return {};
}
std::string shaderCacheSummaryLine() {
  return "[mgpu shadercache] not applicable on web";
}
void shaderCacheNotePipelineMs(double) {}
} // namespace mgpu

extern "C" {
size_t mgpu_shader_cache_dawn_load(const void *, size_t, void *, size_t,
                                   void *) {
  return 0;
}
void mgpu_shader_cache_dawn_store(const void *, size_t, const void *, size_t,
                                  void *) {}

int mgpuShaderCacheSetEnabled(int) { return 0; }
int mgpuShaderCacheSetDirectory(const char *) { return 0; }
int mgpuShaderCacheSetCapBytes(unsigned long long) { return 0; }
int mgpuShaderCacheSetExtraKey(const char *) { return 0; }
int mgpuShaderCacheSetProvider(MGPUShaderCacheLoadFn, MGPUShaderCacheStoreFn,
                               void *) {
  return 0;
}
int mgpuShaderCacheClear(void) { return 0; }
void mgpuGetShaderCacheStats(MGPUShaderCacheStats *out) {
  if (out) std::memset(out, 0, sizeof(*out));
}
int mgpuGetShaderCacheDirectory(char *out, int cap) {
  if (out && cap > 0) out[0] = '\0';
  return 0;
}
} // extern "C"

#endif // __EMSCRIPTEN__

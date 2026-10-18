// Native gate for the port-delivered completions of the external-texture and
// render layers (default D3D11 backend). A stand-in for the Dart VM's API
// table records every post, so each `(token << 1) | ok` is checked for its ok
// bit and for arriving exactly once.
//
// toRGBA: once the async form posts ok its buffer holds exactly what the sync
// form returns, for every input layout (RGBA, BGRA, NV12, GRAY8, I420) and
// with several conversions in flight; the colour setting is read at the call.
// The setting changes NV12, I420 and GRAY8 by the BT.601 / BT.2020 / limited
// range math, whose terms this test derives from each standard's Kr and Kb.
// Refused calls return NULL or -1 and post nothing.
//
// Render pipelines: the settle notification posts ok when a pipeline becomes
// ready, and not ok when its WGSL fails, when it is destroyed while compiling
// or when its context is torn down; one registered after the settle posts at
// once.
//
// The Tier B cross-adapter ring is not exercised: it runs only with Dawn on
// its D3D12 backend, and this suite keeps the D3D11 default.
#include "../include/minigpu.h"
#include "../include/minigpu_external.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

std::atomic<int> errors{0};

void recordLog(int level, const char* message) {
    if (level >= 3) {
        errors.fetch_add(1);
        std::cerr << message << '\n';
    }
    mgpuFreeLogMessage(message);
}

// ---- Stand-in for NativeApi.initializeApiDLData ----------------------------

constexpr int64_t kPort = 0x6d677075; // any id but ILLEGAL_PORT (0)

std::mutex postMutex;
std::condition_variable postArrived;
std::vector<int64_t> posts; // messages posted to kPort, in arrival order
int strayPosts = 0;         // posts to any other port

bool fakePostInteger(int64_t port, int64_t message) {
    {
        std::lock_guard<std::mutex> lock(postMutex);
        if (port == kPort)
            posts.push_back(message);
        else
            ++strayPosts;
    }
    postArrived.notify_all();
    return true;
}

// The layout of the Dart VM's DartApi table (dart_api_dl_impl.h), major 2.
struct FakeEntry { const char* name; void (*function)(void); };
struct FakeApi { int major; int minor; const FakeEntry* functions; };

const FakeEntry kFakeEntries[] = {
    {"Dart_PostInteger", reinterpret_cast<void (*)(void)>(&fakePostInteger)},
    {nullptr, nullptr},
};
FakeApi fakeApi{2, 6, kFakeEntries};

int64_t nextToken = 1;
std::vector<int64_t> postingTokens; // each must post exactly once
std::vector<int64_t> silentTokens;  // each must never post

std::vector<int64_t> postsFor(int64_t token) {
    std::lock_guard<std::mutex> lock(postMutex);
    std::vector<int64_t> found;
    for (int64_t m : posts)
        if ((m >> 1) == token) found.push_back(m);
    return found;
}

// Waits for [token]'s post and returns its ok bit.
bool awaitPost(int64_t token, const char* what) {
    std::unique_lock<std::mutex> lock(postMutex);
    int64_t message = 0;
    auto seen = [&] {
        for (int64_t m : posts) {
            if ((m >> 1) == token) {
                message = m;
                return true;
            }
        }
        return false;
    };
    if (!postArrived.wait_for(lock, std::chrono::seconds(10), seen)) {
        lock.unlock();
        std::cerr << "FAIL: nothing posted: " << what << std::endl;
        std::exit(1);
    }
    return (message & 1) != 0;
}

// The ok bit of the one post [token] made before this call, or fails.
bool postedAtOnce(int64_t token, const char* what) {
    const std::vector<int64_t> found = postsFor(token);
    require(found.size() == 1, what);
    return (found[0] & 1) != 0;
}

// Gives late or duplicate posts time to land, then checks every token.
void expectEachPostedOnce(const char* what) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    for (int64_t token : postingTokens) require(postsFor(token).size() == 1, what);
    for (int64_t token : silentTokens) require(postsFor(token).empty(), what);
    int stray;
    {
        std::lock_guard<std::mutex> lock(postMutex);
        stray = strayPosts;
    }
    require(stray == 0, what);
}

// ---- toRGBA ----------------------------------------------------------------

constexpr uint32_t kW = 34, kH = 18; // not multiples of the 8x8 workgroup
constexpr uint32_t kCw = kW / 2, kCh = kH / 2;

struct Rgba { uint8_t r, g, b, a; };
using Expected = std::function<Rgba(uint32_t, uint32_t)>;

// Deterministic bytes over the whole 0-255 range.
std::vector<uint8_t> pattern(size_t n, uint32_t seed) {
    std::vector<uint8_t> bytes(n);
    uint32_t s = seed;
    for (uint8_t& b : bytes) {
        s = s * 1664525u + 1013904223u;
        b = uint8_t(s >> 24);
    }
    return bytes;
}

MGPUExternalVideoBuffer cpuFrame(MGPUExternalPixelFormat format) {
    MGPUExternalVideoBuffer input{};
    input.content_type = MGPU_EXTERNAL_CONTENT_TYPE_CPU;
    input.pixel_format = format;
    input.width = kW;
    input.height = kH;
    input.fence.sync_fd = -1;
    return input;
}

void setPlane(MGPUExternalVideoBuffer& input, int i, std::vector<uint8_t>& bytes,
              uint32_t w, uint32_t h, uint32_t stride) {
    input.planes[i].data_ptr = bytes.data();
    input.planes[i].width = w;
    input.planes[i].height = h;
    input.planes[i].stride_bytes = stride;
    input.planes[i].dmabuf_fd = -1;
    input.num_planes = uint32_t(i + 1);
}

std::vector<uint8_t> readRgba(MGPUBuffer* out) {
    std::vector<uint8_t> bytes(size_t(kW) * kH * 4);
    mgpuReadSyncUint8(out, bytes.data(), bytes.size(), 0);
    mgpuDestroyBuffer(out);
    return bytes;
}

std::vector<uint8_t> convertSync(MGPUVideoTexture* tex, const std::string& what) {
    MGPUBuffer* out = mgpuVideoTextureToRGBA(tex);
    require(out != nullptr, (what + ": sync toRGBA").c_str());
    return readRgba(out);
}

struct Pending { MGPUBuffer* out; int64_t token; std::string what; };

Pending convertAsync(MGPUVideoTexture* tex, const std::string& what) {
    const int64_t token = nextToken++;
    MGPUBuffer* out = mgpuVideoTextureToRGBAAsyncToPort(tex, kPort, token);
    require(out != nullptr, (what + ": async toRGBA returned NULL").c_str());
    postingTokens.push_back(token);
    return {out, token, what};
}

std::vector<uint8_t> finish(const Pending& p) {
    require(awaitPost(p.token, p.what.c_str()), (p.what + ": async toRGBA posted not ok").c_str());
    return readRgba(p.out);
}

void expectSame(const std::vector<uint8_t>& got, const std::vector<uint8_t>& want,
                const std::string& what) {
    if (got == want) return;
    size_t i = 0;
    while (got[i] == want[i]) ++i;
    std::cerr << "FAIL: " << what << ": pixel (" << i / 4 % kW << ',' << i / 4 / kW
              << ") channel " << i % 4 << " = " << int(got[i]) << ", want "
              << int(want[i]) << std::endl;
    std::exit(1);
}

std::vector<uint8_t> expectedBytes(const Expected& want) {
    std::vector<uint8_t> bytes(size_t(kW) * kH * 4);
    for (uint32_t y = 0; y < kH; ++y) {
        for (uint32_t x = 0; x < kW; ++x) {
            const Rgba c = want(x, y);
            uint8_t* p = &bytes[(size_t(y) * kW + x) * 4];
            p[0] = c.r; p[1] = c.g; p[2] = c.b; p[3] = c.a;
        }
    }
    return bytes;
}

int maxDifference(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    int most = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const int d = std::abs(int(a[i]) - int(b[i]));
        if (d > most) most = d;
    }
    return most;
}

void expectNear(const std::vector<uint8_t>& got, const std::vector<uint8_t>& want,
                int tolerance, const std::string& what) {
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::abs(int(got[i]) - int(want[i])) > tolerance) {
            std::cerr << "FAIL: " << what << ": pixel (" << i / 4 % kW << ','
                      << i / 4 / kW << ") channel " << i % 4 << " = " << int(got[i])
                      << ", want " << int(want[i]) << std::endl;
            std::exit(1);
        }
    }
}

// Kr and Kb of BT.709, BT.601 and BT.2020, in MGPUYuvMatrix order.
const double kKrKb[3][2] = {{0.2126, 0.0722}, {0.299, 0.114}, {0.2627, 0.0593}};

uint8_t truncated(double v) { return uint8_t((v < 0 ? 0 : v > 1 ? 1 : v) * 255.0); }
uint8_t rounded(double v) { return uint8_t((v < 0 ? 0 : v > 1 ? 1 : v) * 255.0 + 0.5); }

// The documented conversion, truncated like the shader: full range centres
// chroma on 0.5; limited range maps luma 16-235 and chroma 16-240.
Rgba yuv(uint8_t yb, uint8_t ub, uint8_t vb, int matrix, bool full) {
    const double kr = kKrKb[matrix][0], kb = kKrKb[matrix][1], kg = 1 - kr - kb;
    double y, cb, cr;
    if (full) {
        y = yb / 255.0;
        cb = ub / 255.0 - 0.5;
        cr = vb / 255.0 - 0.5;
    } else {
        y = (yb - 16) / 219.0;
        cb = (ub - 128) / 224.0;
        cr = (vb - 128) / 224.0;
    }
    return {truncated(y + 2 * (1 - kr) * cr),
            truncated(y - 2 * kb * (1 - kb) / kg * cb - 2 * kr * (1 - kr) / kg * cr),
            truncated(y + 2 * (1 - kb) * cb), 255};
}

void testToRgba() {
    const int errorsBefore = errors.load();
    const size_t n = size_t(kW) * kH;
    std::vector<uint8_t> rgba = pattern(n * 4, 1), bgra = pattern(n * 4, 2);
    std::vector<uint8_t> gray = pattern(n, 3);
    std::vector<uint8_t> nvY = pattern(n, 4), nvUV = pattern(size_t(kW) * kCh, 5);
    std::vector<uint8_t> yp = pattern(n, 6), up = pattern(size_t(kCw) * kCh, 7),
                         vp = pattern(size_t(kCw) * kCh, 8);

    MGPUExternalVideoBuffer input = cpuFrame(MGPU_EXTERNAL_PIXEL_FORMAT_RGBA32);
    setPlane(input, 0, rgba, kW, kH, kW * 4);
    MGPUVideoTexture* rgbaTex = mgpuImportVideoFrame(&input);
    input = cpuFrame(MGPU_EXTERNAL_PIXEL_FORMAT_BGRA32);
    setPlane(input, 0, bgra, kW, kH, kW * 4);
    MGPUVideoTexture* bgraTex = mgpuImportVideoFrame(&input);
    input = cpuFrame(MGPU_EXTERNAL_PIXEL_FORMAT_GRAY8);
    setPlane(input, 0, gray, kW, kH, kW);
    MGPUVideoTexture* grayTex = mgpuImportVideoFrame(&input);
    input = cpuFrame(MGPU_EXTERNAL_PIXEL_FORMAT_NV12);
    setPlane(input, 0, nvY, kW, kH, kW);
    setPlane(input, 1, nvUV, kCw, kCh, kW);
    MGPUVideoTexture* nv12Tex = mgpuImportVideoFrame(&input);
    input = cpuFrame(MGPU_EXTERNAL_PIXEL_FORMAT_YUV420P_AS_RGB_PLANES);
    setPlane(input, 0, yp, kW, kH, kW);
    setPlane(input, 1, up, kCw, kCh, kCw);
    setPlane(input, 2, vp, kCw, kCh, kCw);
    MGPUVideoTexture* i420Tex = mgpuImportVideoFrame(&input);
    require(rgbaTex && bgraTex && grayTex && nv12Tex && i420Tex, "CPU imports");

    auto nv12Want = [&](int matrix, bool full) -> Expected {
        return [&, matrix, full](uint32_t x, uint32_t y) {
            const uint8_t* uv = &nvUV[size_t(y / 2) * kW + (x / 2) * 2];
            return yuv(nvY[size_t(y) * kW + x], uv[0], uv[1], matrix, full);
        };
    };
    auto i420Want = [&](int matrix, bool full) -> Expected {
        return [&, matrix, full](uint32_t x, uint32_t y) {
            const size_t c = size_t(y / 2) * kCw + x / 2;
            return yuv(yp[size_t(y) * kW + x], up[c], vp[c], matrix, full);
        };
    };

    // Every layout in flight at once; each async result is the sync result,
    // and that is the documented conversion of the source.
    struct Case { const char* name; MGPUVideoTexture* tex; std::vector<uint8_t> want; int tolerance; };
    Case cases[] = {
        {"RGBA", rgbaTex, rgba, 0},
        {"BGRA", bgraTex, expectedBytes([&](uint32_t x, uint32_t y) {
             const uint8_t* p = &bgra[(size_t(y) * kW + x) * 4];
             return Rgba{p[2], p[1], p[0], p[3]};
         }), 0},
        {"NV12", nv12Tex, expectedBytes(nv12Want(MGPU_YUV_MATRIX_BT709, true)), 2},
        {"GRAY8", grayTex, expectedBytes([&](uint32_t x, uint32_t y) {
             const uint8_t v = gray[size_t(y) * kW + x];
             return Rgba{v, v, v, 255};
         }), 0},
        {"I420", i420Tex, expectedBytes(i420Want(MGPU_YUV_MATRIX_BT709, true)), 2},
    };
    std::vector<Pending> pending;
    for (const Case& c : cases) pending.push_back(convertAsync(c.tex, c.name));
    std::vector<std::vector<uint8_t>> defaults;
    for (size_t i = 0; i < pending.size(); ++i) {
        const std::string name = cases[i].name;
        std::vector<uint8_t> got = finish(pending[i]);
        expectSame(got, convertSync(cases[i].tex, name), name + ": async equals sync");
        expectNear(got, cases[i].want, cases[i].tolerance, name + ": default conversion");
        defaults.push_back(std::move(got));
    }
    std::cout << "PASS async toRGBA equals sync for RGBA, BGRA, NV12, GRAY8, I420 (5 in flight)"
              << std::endl;

    // The setting is read at the call: a change right after it does not
    // reach the conversion already made.
    Pending early = convertAsync(nv12Tex, "NV12 before a colour change");
    require(mgpuVideoTextureSetYuvColor(nv12Tex, MGPU_YUV_MATRIX_BT601, 0) == 0,
            "set BT.601 limited");
    expectSame(finish(early), defaults[2], "NV12: a colour change after the call");

    struct Setting { int matrix; bool full; const char* name; };
    const Setting settings[] = {
        {MGPU_YUV_MATRIX_BT709, true, "BT.709 full"},
        {MGPU_YUV_MATRIX_BT601, true, "BT.601 full"},
        {MGPU_YUV_MATRIX_BT709, false, "BT.709 limited"},
        {MGPU_YUV_MATRIX_BT601, false, "BT.601 limited"},
        {MGPU_YUV_MATRIX_BT2020, false, "BT.2020 limited"},
    };
    // A setting the library ignored must not hide inside the tolerance: every
    // pair of settings moves some pixel well past it.
    std::vector<std::vector<uint8_t>> nv12Expected, i420Expected;
    for (const Setting& s : settings) {
        nv12Expected.push_back(expectedBytes(nv12Want(s.matrix, s.full)));
        i420Expected.push_back(expectedBytes(i420Want(s.matrix, s.full)));
    }
    for (size_t a = 0; a < nv12Expected.size(); ++a)
        for (size_t b = a + 1; b < nv12Expected.size(); ++b)
            require(maxDifference(nv12Expected[a], nv12Expected[b]) > 8 &&
                        maxDifference(i420Expected[a], i420Expected[b]) > 8,
                    "fixture: two colour settings give nearly the same pixels");

    std::vector<uint8_t> lastNv12;
    for (size_t i = 0; i < sizeof(settings) / sizeof(settings[0]); ++i) {
        const Setting& s = settings[i];
        const int full = s.full ? 1 : 0;
        require(mgpuVideoTextureSetYuvColor(nv12Tex, s.matrix, full) == 0 &&
                    mgpuVideoTextureSetYuvColor(i420Tex, s.matrix, full) == 0,
                "a known colour setting is refused");
        const std::string name = s.name;
        Pending nv = convertAsync(nv12Tex, "NV12 " + name);
        Pending i4 = convertAsync(i420Tex, "I420 " + name);
        std::vector<uint8_t> nvGot = finish(nv), i4Got = finish(i4);
        expectSame(nvGot, convertSync(nv12Tex, name), "NV12 " + name + ": async equals sync");
        expectSame(i4Got, convertSync(i420Tex, name), "I420 " + name + ": async equals sync");
        expectNear(nvGot, nv12Expected[i], 2, "NV12 " + name);
        expectNear(i4Got, i420Expected[i], 2, "I420 " + name);
        if (i == 0) expectSame(nvGot, defaults[2], "explicit BT.709 full equals the default");
        lastNv12 = std::move(nvGot);
    }

    // GRAY8 takes the range only; the packed formats take neither.
    require(mgpuVideoTextureSetYuvColor(grayTex, MGPU_YUV_MATRIX_BT601, 0) == 0,
            "GRAY8 limited range refused");
    std::vector<uint8_t> grayLimited = finish(convertAsync(grayTex, "GRAY8 limited"));
    expectSame(grayLimited, convertSync(grayTex, "GRAY8 limited"), "GRAY8 limited: async equals sync");
    // (y - 16) * 255 / 219 never lands on a .5, so the rounding is exact.
    expectNear(grayLimited, expectedBytes([&](uint32_t x, uint32_t y) {
                   const uint8_t v = rounded((gray[size_t(y) * kW + x] - 16) / 219.0);
                   return Rgba{v, v, v, 255};
               }), 0, "GRAY8 limited range stretches 16-235");
    require(mgpuVideoTextureSetYuvColor(grayTex, MGPU_YUV_MATRIX_BT2020, 0) == 0,
            "GRAY8 BT.2020 refused");
    expectSame(convertSync(grayTex, "GRAY8 BT.2020"), grayLimited, "GRAY8 ignores the matrix");
    require(mgpuVideoTextureSetYuvColor(bgraTex, MGPU_YUV_MATRIX_BT601, 0) == 0,
            "BGRA colour setting refused");
    expectSame(convertSync(bgraTex, "BGRA"), defaults[1], "BGRA ignores the colour setting");

    // Refusals change nothing and post nothing.
    require(mgpuVideoTextureSetYuvColor(nullptr, MGPU_YUV_MATRIX_BT709, 1) == -1,
            "a NULL texture's colour setting is refused");
    require(mgpuVideoTextureSetYuvColor(nv12Tex, 3, 1) == -1 &&
                mgpuVideoTextureSetYuvColor(nv12Tex, -1, 0) == -1,
            "an unknown matrix is refused");
    expectSame(convertSync(nv12Tex, "NV12 after refusals"), lastNv12,
               "a refused colour setting changes nothing");
    const int64_t refused = nextToken++;
    require(mgpuVideoTextureToRGBAAsyncToPort(nullptr, kPort, refused) == nullptr,
            "async toRGBA of NULL returns NULL");
    silentTokens.push_back(refused);

    mgpuDestroyVideoTexture(rgbaTex);
    mgpuDestroyVideoTexture(bgraTex);
    mgpuDestroyVideoTexture(grayTex);
    mgpuDestroyVideoTexture(nv12Tex);
    mgpuDestroyVideoTexture(i420Tex);
    mgpuDrainWorkQueue();
    expectEachPostedOnce("toRGBA: a completion posted other than exactly once");
    require(errors.load() == errorsBefore, "error logged during the toRGBA tests");
    std::cout << "PASS colour settings (BT.601/709/2020, full/limited; GRAY8 range only), "
                 "read at the call; refusals post nothing" << std::endl;
}

// ---- Render pipeline settle ------------------------------------------------

// The copy pipeline of the render contract test: one unfilterable 2D texture
// at fragment slot 0, an RGBA8 target.
const char* kCopyWgsl = R"WGSL(
@group(0) @binding(0) var source: texture_2d<f32>;
@vertex fn vsMain(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fsMain(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
  return textureLoad(source, vec2<i32>(at.xy), 0);
}
)WGSL";

MGPURenderPipeline* createPipeline(const std::string& wgsl) {
    const std::vector<uint32_t> d = {1, 1, 0, 0, 0, 0, 0, 0, 4, 0, 1, 0, 3, 2, 0, 0};
    MGPURenderPipeline* p =
        mgpuCreateRenderPipeline(wgsl.c_str(), "vsMain", "fsMain", d.data(), (int)d.size());
    require(p != nullptr, "pipeline allocation failed");
    return p;
}

int64_t notifySettled(MGPURenderPipeline* p) {
    const int64_t token = nextToken++;
    mgpuRenderPipelineNotifySettledToPort(p, kPort, token);
    postingTokens.push_back(token);
    return token;
}

// Returns the ready pipeline, kept for the teardown case.
MGPURenderPipeline* testRenderSettle() {
    MGPURenderPipeline* ready = createPipeline(kCopyWgsl);
    require(awaitPost(notifySettled(ready), "ready pipeline"), "a ready pipeline posts not ok");
    require(mgpuRenderPipelineStatus(ready) == 1, "status after an ok post is not 1");
    require(postedAtOnce(notifySettled(ready), "a waiter after the settle did not post at once"),
            "a waiter after a ready settle posts not ok");

    std::string typo = kCopyWgsl;
    typo.replace(typo.find("textureLoad(source"), 18, "textureLoad(sauce");
    MGPURenderPipeline* broken = createPipeline(typo);
    require(!awaitPost(notifySettled(broken), "WGSL error"), "invalid WGSL posts ok");
    require(mgpuRenderPipelineStatus(broken) == 2, "status after invalid WGSL is not 2");
    require(!postedAtOnce(notifySettled(broken), "a waiter after a failure did not post at once"),
            "a waiter after a failed settle posts ok");
    mgpuDestroyRenderPipeline(broken);
    require(!postedAtOnce(notifySettled(nullptr), "NULL did not post at once"), "NULL posts ok");
    mgpuDrainWorkQueue();
    // The WGSL error above is logged as a validation error; nothing below may.
    const int errorsBefore = errors.load();

    // Destroyed while compiling: the destroy itself posts not ok. A large
    // upload holds the worker so the compile queues behind it; if the compile
    // still settles first its post is ok, and the case runs again.
    const int side = 4096;
    std::vector<uint8_t> ballastBytes(size_t(side) * side * 4, 7);
    bool closedFirst = false;
    for (int attempt = 0; attempt < 5 && !closedFirst; ++attempt) {
        MGPUTexture* ballast = mgpuCreateTexture2D(side, side, 1, 2 | 4, 1, 1);
        require(ballast != nullptr, "ballast allocation failed");
        require(mgpuWriteTexture2D(ballast, ballastBytes.data(), (int)ballastBytes.size(), 0, 0) == 0,
                "ballast upload refused");
        MGPURenderPipeline* p = createPipeline(kCopyWgsl);
        const int64_t token = notifySettled(p);
        mgpuDestroyRenderPipeline(p);
        const std::vector<int64_t> found = postsFor(token);
        if (found.size() == 1 && (found[0] & 1) == 0)
            closedFirst = true;
        else
            require(awaitPost(token, "compiled before its destroy"),
                    "a pipeline compiled before its destroy posts not ok");
        mgpuDrainWorkQueue(); // the queued compile now finds it closed
        mgpuDestroyTexture(ballast);
    }
    require(closedFirst, "a destroy never settled a compiling pipeline");
    mgpuDrainWorkQueue();
    expectEachPostedOnce("render: a settle posted other than exactly once");
    require(errors.load() == errorsBefore, "error logged during the settle tests");
    std::cout << "PASS render settle: ready ok, WGSL error not ok, destroy while compiling "
                 "not ok, late waiter and NULL at once" << std::endl;
    return ready;
}

// A context teardown closes its pipelines: a waiter registered before posts
// once (ok when the compile beat the teardown), the pipelines then read 3,
// and a waiter registered after posts not ok at once.
void testTeardownSettle(MGPURenderPipeline* ready) {
    MGPURenderPipeline* compiling = createPipeline(kCopyWgsl);
    const int64_t token = notifySettled(compiling);
    mgpuDestroyContext();
    awaitPost(token, "a waiter across its context teardown");
    require(mgpuRenderPipelineStatus(compiling) == 3 && mgpuRenderPipelineStatus(ready) == 3,
            "a pipeline of a torn-down context does not read 3");
    require(!postedAtOnce(notifySettled(compiling), "a waiter after a teardown did not post at once") &&
                !postedAtOnce(notifySettled(ready), "a waiter after a teardown did not post at once"),
            "a waiter after a teardown posts ok");
    mgpuDestroyRenderPipeline(compiling);
    mgpuDestroyRenderPipeline(ready);
    expectEachPostedOnce("teardown: a settle posted other than exactly once");
    std::cout << "PASS render settle across a context teardown" << std::endl;
}

} // namespace

int main() {
    mgpuSetLogCallback(recordLog);
    require(mgpuInitDartApi(&fakeApi) == 0, "the stand-in Dart API was refused");
    mgpuInitializeContext();
    testToRgba();
    testTeardownSettle(testRenderSettle());
    std::cout << "SKIP Tier B ring: runs only with Dawn on its D3D12 backend; this suite "
                 "keeps the D3D11 default (MGPU_BACKEND unset)" << std::endl;
    mgpuSetLogCallback(nullptr);
    std::cout << "External async test passed." << std::endl;
    return 0;
}

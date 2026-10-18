// Production owner with a CPU fake registrar/provider. No Flutter engine, GPU,
// capture, actual shared image or qualified deployed profile is initialized.
#include "../include/minigpu_view/minigpu_view_private_preview.h"
#include "../include/minigpu_view/minigpu_view_private_dispatch.h"
#include "../private_preview_owner.h"
#include "../private_preview_dispatcher.h"
#include "../private_output_api.h"
#include "../registrar_association.h"
#include "../include/minigpu_view/minigpu_view_private_owner_close.h"
#include <windows.h>
#include <array>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
using namespace minigpu_view::private_preview;
extern "C" int private_preview_c_abi(void);
extern "C" int private_preview_native_busy(void);
extern "C" int private_preview_native_unsupported(void);
static void require(bool ok, const char *message) {
  if (!ok) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); }
}
class Raster {
  std::mutex gate;
  std::condition_variable changed;
  std::deque<std::function<void()>> tasks;
  bool stop = false;
  std::thread worker;
public:
  Raster() : worker([this] {
    for (;;) {
      std::function<void()> run;
      {
        std::unique_lock<std::mutex> lock(gate);
        changed.wait(lock, [this] { return stop || !tasks.empty(); });
        if (stop && tasks.empty()) return;
        run = std::move(tasks.front()); tasks.pop_front();
      }
      run();
    }
  }) {}
  ~Raster() {
    { std::lock_guard<std::mutex> lock(gate); stop = true; }
    changed.notify_one(); worker.join();
  }
  HANDLE handle() { return worker.native_handle(); }
  void run(std::function<void()> task) {
    std::promise<void> done; auto ready = done.get_future();
    {
      std::lock_guard<std::mutex> lock(gate);
      tasks.push_back([&] { task(); done.set_value(); });
    }
    changed.notify_one(); ready.get();
  }
};
class Registrar : public flutter::TextureRegistrar {
public:
  enum Mode { normal, delayed, inlineFailure, wrongThread, never } mode = normal;
  Raster &raster;
  std::map<int64_t, flutter::TextureVariant *> textures;
  std::function<void()> pending;
  std::function<void()> onRegister;
  int registrations = 0, removals = 0;
  int64_t next = 1, removing = -1;
  bool failRegistration = false, failNotification = false, actualRemoval = false;
  explicit Registrar(Raster &r) : raster(r) {}
  int64_t RegisterTexture(flutter::TextureVariant *v) override {
    require(std::get<flutter::GpuSurfaceTexture>(*v).ObtainDescriptor(16, 8) == nullptr,
            "registration placeholder gate is closed before loan acquisition");
    ++registrations;
    if (failRegistration) return -1;
    if (onRegister) onRegister();
    textures[next] = v; return next++;
  }
  bool MarkTextureFrameAvailable(int64_t) override { return !failNotification; }
  void UnregisterTexture(int64_t id, std::function<void()> cb) override {
    ++removals;
    if (mode == normal) {
      raster.run([&] { textures.erase(id); actualRemoval = true; cb(); });
    } else if (mode == delayed) { removing = id; pending = std::move(cb); }
    else if (mode == inlineFailure) { cb(); } // Posting failed: still registered.
    else if (mode == wrongThread) { std::thread wrong(std::move(cb)); wrong.join(); }
  }
  bool UnregisterTexture(int64_t) override { require(false, "legacy overload unused"); return false; }
  const FlutterDesktopGpuSurfaceDescriptor *frame(int64_t id) {
    const FlutterDesktopGpuSurfaceDescriptor *result = nullptr;
    raster.run([&] { result = std::get<flutter::GpuSurfaceTexture>(*textures.at(id))
        .ObtainDescriptor(16, 8); });
    return result;
  }
  void removePending() {
    raster.run([&] {
      textures.erase(removing); actualRemoval = true;
      auto cb = std::move(pending); cb();
    });
  }
};
struct FakeOutput {
  OutputInfo info{};
  Registrar *registrar;
  uint64_t loanId;
  uint32_t live = 0, consumes = 0, releases = 0, acquires = 0;
  uint32_t loanOperations = 0, loanBorrows = 0;
  bool sealed = false, quarantined = false, acquired = false;
  bool failAcquire = false, failConsume = false, failRelease = false;
  bool nextBorrowBusy = false;
  std::function<void()> onQuery, onAcquire;
  explicit FakeOutput(Registrar &r) : registrar(&r) {
    static uint64_t ids = 10;
    info = {80,1,2,2,0,0,ids++,1,2,16*8*4,16,8,0,0,0};
    loanId = info.id + 1000;
  }
};
struct FakeBorrow { FakeOutput *output; };
extern "C" __declspec(dllexport) int mgpuExternalPoolQueryCapabilities(void *, void *, uint32_t) { return 0; }
extern "C" __declspec(dllexport) int mgpuPrivateOutputGetInfo(void *p, OutputInfo *out, uint32_t size) {
  if (!p || !out || size < 80) return -1;
  auto &f = *static_cast<FakeOutput *>(p); *out = f.info;
  if (f.onQuery) f.onQuery();
  return 0;
}
extern "C" __declspec(dllexport) int mgpuPrivateOutputAcquirePreviewLoan(void *p, void **out) {
  auto &f = *static_cast<FakeOutput *>(p); *out = nullptr; ++f.acquires;
  require(!f.registrar->textures.empty(), "real registration exists before native loan");
  if (f.failAcquire || f.acquired) return static_cast<int>(NativeResult::wouldBlock);
  f.acquired = true; *out = p;
  if (f.onAcquire) f.onAcquire();
  return 0;
}
extern "C" __declspec(dllexport) int mgpuPrivatePreviewLoanGetInfo(void *p, OutputInfo *out, uint32_t size) {
  if (!p || !out || size < 80) return -1;
  auto &f = *static_cast<FakeOutput *>(p); *out = f.info;
  out->kind = 3; out->id = f.loanId; out->parent = f.info.id;
  out->operations = f.loanOperations; out->borrows = f.loanBorrows; return 0;
}
extern "C" __declspec(dllexport) int mgpuPrivatePreviewBorrowDescriptor(
    void *p, void **native, BorrowInfo *out, uint32_t size) {
  auto &f = *static_cast<FakeOutput *>(p); *native = nullptr;
  if (!out || size < 64) return -1;
  if (f.nextBorrowBusy) {
    f.nextBorrowBusy = false; return static_cast<int>(NativeResult::wouldBlock);
  }
  if (f.sealed || f.quarantined) return -3;
  *native = new FakeBorrow{&f}; ++f.live;
  static uint64_t ids = 1;
  *out = {64,1,f.info.id,f.loanId,ids++,16,8,1,0,0x1234,2}; return 0;
}
extern "C" __declspec(dllexport) int mgpuPrivatePreviewBorrowConsumed(void *p) {
  auto *b = static_cast<FakeBorrow *>(p); auto &f = *b->output;
  ++f.consumes; if (f.failConsume) return -5;
  --f.live; delete b; return 0;
}
extern "C" __declspec(dllexport) int mgpuPrivatePreviewLoanSeal(void *p) {
  static_cast<FakeOutput *>(p)->sealed = true; return 0;
}
extern "C" __declspec(dllexport) int mgpuPrivatePreviewLoanQuarantine(void *p) {
  static_cast<FakeOutput *>(p)->quarantined = true; return -7;
}
extern "C" __declspec(dllexport) int mgpuPrivatePreviewLoanReleaseAfterRemoval(void *p, uint64_t id) {
  auto &f = *static_cast<FakeOutput *>(p); ++f.releases;
  require(f.registrar->actualRemoval && f.sealed && f.live == 0 && id == f.loanId,
          "native final release follows actual removal, seal and borrow drain");
  return f.failRelease ? -5 : 0;
}
static bool qualified = true;
static int verify(void *, const MGPUViewHostBinding *, void *) { return qualified ? 0 : -4; }
static MGPUViewPrivateRegistration *add(MGPUViewTrustedHost *host, FakeOutput *f,
                                       MGPUViewPrivateInfo &out, int expected = 0) {
  MGPUViewPrivateRegistration *reg = nullptr;
  require(MGPUViewRegisterPrivateOutput(host, f, f->info.id, &reg, &out, sizeof(out)) == expected && reg,
          "accepted registration returns its owner even on post-registration error");
  return reg;
}
static void collect(MGPUViewPrivateRegistration *reg) {
  MGPUViewPrivateReceipt receipt{};
  require(MGPUViewCollectPrivateRetirement(reg, &receipt, sizeof(receipt)) == 0 &&
              receipt.struct_size == 48 && receipt.version == 1 && receipt.registration_id &&
              receipt.engine_generation == 9 && receipt.output_id,
          "versioned identity-bound ownership receipt");
  MGPUViewPrivateInfo info{};
  require(MGPUViewGetPrivateRegistrationInfo(reg, &info, sizeof(info)) == 0 &&
              !info.registered && info.phase == MGPU_VIEW_RELEASED,
          "released status does not claim a historical texture is registered");
  require(MGPUViewReleasePrivateRegistration(reg) == 0, "retired registration releases charge");
}
#include "private_preview_dispatch_test.inc"
#include "registrar_association_test.inc"
#include "private_owner_close_test.inc"
int main() {
  require(private_preview_c_abi(), "plain C layout and cold refusal");
  require(static_cast<int>(NativeResult::wouldBlock) == private_preview_native_busy() &&
              static_cast<int>(NativeResult::unsupported) == private_preview_native_unsupported(),
          "fake producer result values conform to the actual native C header");
  Raster raster; Registrar registrar(raster);
  MGPUViewHostBinding binding{80,1,1,0,9,77,
    reinterpret_cast<void *>(&verify), reinterpret_cast<void *>(&mgpuExternalPoolQueryCapabilities),
    GetCurrentThread(), raster.handle(), verify, nullptr};
  MGPUViewTrustedHost *host = nullptr;
  qualified = false;
  require(MGPUViewBindTrustedHost(&registrar, &binding, &host) == -4 && !host,
          "unqualified engine refuses before registration or loan");
  qualified = true;
  auto wrong = binding; wrong.original_minigpu_query = reinterpret_cast<void *>(&verify);
  require(MGPUViewBindTrustedHost(&registrar, &wrong, &host) == -4 && !host,
          "wrong export anchor refuses");
  auto merged = binding; merged.raster_thread_handle = GetCurrentThread();
  require(MGPUViewBindTrustedHost(&registrar, &merged, &host) == -4 && !host,
          "merged platform/raster profile refuses");
  require(MGPUViewBindTrustedHost(&registrar, &binding, &host) == 0 && host,
          "explicit CPU fake engine qualification binds actual production owner");
  MGPUViewPrivateInfo details{};
  associationCases(binding, registrar, raster);
  ownerCloseCases(binding, registrar);
  dispatchCases(host, binding, registrar, raster);
  registrar.registrations = registrar.removals = 0;
  for (int field = 0; field != 3; ++field) {
    FakeOutput invalid(registrar);
    if (field == 0) invalid.info.operations = 1;
    if (field == 1) invalid.info.borrows = 1;
    if (field == 2) invalid.info.parent = 1;
    MGPUViewPrivateRegistration *rejected = nullptr;
    require(MGPUViewRegisterPrivateOutput(host, &invalid, invalid.info.id,
        &rejected, &details, sizeof(details)) == -1 && !rejected &&
        !invalid.acquires && !registrar.registrations,
        "output admission rejects pending counters/foreign parent before registration");
  }
  auto *f = new FakeOutput(registrar); auto *reg = add(host, f, details);
  f->nextBorrowBusy = true;
  require(!registrar.frame(details.texture_id) && !f->quarantined &&
              MGPUViewGetPrivateRegistrationInfo(reg, &details, sizeof(details)) == 0 &&
              details.phase == MGPU_VIEW_ACTIVE,
          "native transient busy drops a descriptor without quarantining the view");
  std::array<const FlutterDesktopGpuSurfaceDescriptor *,4> frames{};
  for (auto &p : frames) { p = registrar.frame(details.texture_id); require(p, "descriptor borrow"); }
  require(f->live == 4 && !registrar.frame(details.texture_id), "four descriptor bound, fifth drops");
  const auto oldContext = frames[0]->release_context;
  const auto oldCallback = frames[0]->release_callback;
  raster.run([&] { oldCallback(oldContext); oldCallback(oldContext); });
  require(f->live == 3 && f->consumes == 1 && frames[0]->visible_width == 16 &&
              frames[0]->visible_height == 8, "binding-failure release preserves descriptor storage");
  frames[0] = registrar.frame(details.texture_id);
  raster.run([&] { oldCallback(oldContext); });
  require(f->live == 4 && f->consumes == 1, "stale generation cannot consume replacement borrow");
  registrar.mode = Registrar::delayed;
  require(MGPUViewBeginPrivateRetirement(reg) == 0 &&
              MGPUViewBeginPrivateRetirement(reg) == -6 && registrar.removals == 1 &&
              !registrar.frame(details.texture_id), "retire closes gate, seals once, unregisters once");
  MGPUViewPrivateReceipt receipt{};
  require(MGPUViewCollectPrivateRetirement(reg, &receipt, sizeof(receipt)) == -6,
          "no callback cannot manufacture a removal receipt");
  registrar.removePending();
  require(MGPUViewGetPrivateRegistrationInfo(reg, &details, sizeof(details)) == 0 &&
              !details.registered && details.phase == MGPU_VIEW_REMOVED,
          "confirmed removal updates current registration status");
  require(MGPUViewCollectPrivateRetirement(reg, &receipt, sizeof(receipt)) == -6 && f->releases == 0,
          "actual removal still waits for descriptor consumption");
  raster.run([&] { for (auto p : frames) p->release_callback(p->release_context); });
  require(f->live == 0, "all borrow receipts drained");
  collect(reg); require(f->releases == 1, "one checked native final release");

  registrar.mode = Registrar::normal;
  auto *empty = new FakeOutput(registrar); empty->failAcquire = true;
  reg = add(host, empty, details, -6);
  require(empty->acquires == 1 && !empty->acquired && MGPUViewBeginPrivateRetirement(reg) == 0,
          "failed acquire leaves only closed empty registration");
  collect(reg);
  auto *unused = new FakeOutput(registrar);
  reg = add(host, unused, details);
  require(MGPUViewBeginPrivateRetirement(reg) == 0, "never-sampled removal callback before return");
  collect(reg);
  registrar.failRegistration = true;
  auto *unregistered = new FakeOutput(registrar);
  reg = reinterpret_cast<MGPUViewPrivateRegistration *>(uintptr_t(1));
  require(MGPUViewRegisterPrivateOutput(host, unregistered, unregistered->info.id,
      &reg, &details, sizeof(details)) == -5 && !reg && !unregistered->acquires,
      "known registration refusal acquires no loan and charges no registration");
  registrar.failRegistration = false;
  registrar.failNotification = true;
  auto *unnotified = new FakeOutput(registrar); reg = add(host, unnotified, details, -5);
  require(!registrar.frame(details.texture_id) && MGPUViewBeginPrivateRetirement(reg) == 0,
          "failed notification returns drainable closed owner");
  collect(reg); registrar.failNotification = false;

  for (int field = 0; field != 2; ++field) {
    auto *invalid = new FakeOutput(registrar);
    if (field == 0) invalid->loanOperations = 1;
    else invalid->loanBorrows = 1;
    reg = add(host, invalid, details, -7);
    require(invalid->acquires == 1 && invalid->quarantined &&
        MGPUViewCollectPrivateRetirement(reg, &receipt, sizeof(receipt)) == -7 &&
        !invalid->releases,
        "pending loan work retains the admitted registration and native owner");
  }

  for (const auto mode : {Registrar::inlineFailure, Registrar::wrongThread}) {
    registrar.mode = mode;
    auto *ambiguous = new FakeOutput(registrar); reg = add(host, ambiguous, details);
    require(MGPUViewBeginPrivateRetirement(reg) == -7 &&
        MGPUViewCollectPrivateRetirement(reg, &receipt, sizeof(receipt)) == -7 &&
        MGPUViewReleasePrivateRegistration(reg) == -6 && ambiguous->quarantined &&
        !ambiguous->releases, "ambiguous callbacks quarantine, never grant native release");
  }
  registrar.mode = Registrar::normal;
  auto *badRelease = new FakeOutput(registrar); badRelease->failRelease = true;
  reg = add(host, badRelease, details);
  require(MGPUViewBeginPrivateRetirement(reg) == 0 &&
      MGPUViewCollectPrivateRetirement(reg, &receipt, sizeof(receipt)) == -7 &&
      MGPUViewCollectPrivateRetirement(reg, &receipt, sizeof(receipt)) == -7 &&
      badRelease->releases == 1, "failed native release quarantines without destructive retry");
  registrar.mode = Registrar::never;
  auto *shutdown = new FakeOutput(registrar); reg = add(host, shutdown, details);
  require(MGPUViewBeginPrivateRetirement(reg) == 0 &&
      MGPUViewCollectPrivateRetirement(reg, &receipt, sizeof(receipt)) == -6,
      "missing removal callback remains pending");
  minigpu_view::ClosePrivatePreviewRegistrar(&registrar);
  require(MGPUViewCollectPrivateRetirement(reg, &receipt, sizeof(receipt)) == -7 &&
      MGPUViewReleaseTrustedHost(host) == -6, "plugin shutdown retains unresolved host and loans");
  MGPUViewPrivateStats stats{};
  require(MGPUViewGetPrivateStats(&stats, sizeof(stats)) == 0 && stats.struct_size == 40 &&
      stats.hosts == 2 && stats.registrations == 8 && stats.quarantined_registrations == 8 &&
      !stats.descriptor_borrows && stats.max_hosts == 16 && stats.max_registrations == 256 &&
      stats.max_descriptors == 4, "only unresolved owners remain charged in bounded quarantine");
  std::cout << "PASS private preview production owner with CPU fake registrar/provider: "
      "closed registration, bounded descriptor generations/storage, checked removal, "
      "source-fenced platform dispatch/callback return, reentrant shutdown quarantine; "
      "no engine/GPU/capture qualification\n";
}

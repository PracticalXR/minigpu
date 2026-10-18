// Actual non-test-macro production factories; no native context/engine/GPU.
#include "../../include/minigpu_capture_source_owner.h"
#include "../../include/minigpu_setup_layout.h"
#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <new>
#include <stdexcept>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <malloc.h>
#endif

#include "setup_allocation_probe.h"

namespace {
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
[[maybe_unused]] void checkRequest(size_t index, const MGPUCaptureSetupExtentV1 &expected) {
  const auto &entry = probe::state.entries[index];
  require(entry.bytes == expected.bytes && entry.address % expected.alignment == 0,
      "actual factory allocation differs from original DLL's exact block/object extent");
  require(expected.alignment > __STDCPP_DEFAULT_NEW_ALIGNMENT__ ? entry.alignmentArgument == expected.alignment :
      entry.alignmentArgument == 0, "new dispatch/alignment argument differs from actual type requirement");
}
[[maybe_unused]] void factories(const MGPUCaptureSetupLayoutV1 &layout) {
  using namespace mgpu::source_scope;
  probe::state = {}; probe::state.active = true;
  SourceRootOwner root;
  const auto result = SourceRoot::create({10, 9, 1, 0}, root);
  probe::state.active = false;
  require(result == Result::Ok && root && !probe::state.overflow && probe::state.count == 2,
      "production SourceRoot create does not have exactly its two declared setup allocations");
  checkRequest(0, layout.kinds[MGPU_CAPTURE_SETUP_ROOT_STATE]);
  checkRequest(1, layout.kinds[MGPU_CAPTURE_SETUP_ROOT_BOX]);
  auto alias = root->domain();
  JoinReceipt receipt;
  require(root->seal() == Result::Ok && root->tryJoin(receipt) == Result::Ok &&
      root->release(receipt) == Result::Ok, "real root joined release refused");
  root.reset();
  require(probe::state.entries[1].freed && !probe::state.entries[0].freed,
      "root join/wrapper disposal incorrectly treated a retained State alias as deallocated");
  alias = {};
  require(probe::state.entries[0].freed, "actual last State alias did not deallocate its shared block");

  const auto before = mgpu::capture_source::liveOwners();
  probe::state = {};
  mgpu::capture_source::Owner held;
  {
    mgpu::capture_source::PreparedSource prepared;
    MGPUCaptureSourceDesc desc{48, 1, 10, 9, 1, 0, 17, 19, 360};
    probe::state.active = true;
    const auto preparedResult = prepared.prepare(desc);
    probe::state.active = false;
    require(preparedResult == 0 && !probe::state.overflow && probe::state.count == 2,
        "production PreparedSource prepare does not have its two declared setup allocations");
    checkRequest(0, layout.kinds[MGPU_CAPTURE_SETUP_PREPARED_STATE]);
    checkRequest(1, layout.kinds[MGPU_CAPTURE_SETUP_PREPARED_ANCHOR]);
    held = prepared.owner();
  }
  require(probe::state.entries[1].freed && !probe::state.entries[0].freed &&
      mgpu::capture_source::liveOwners() == before + 1, "prepared owner alias lost actual State/control-block custody");
  held.reset();
  require(probe::state.entries[0].freed && mgpu::capture_source::liveOwners() == before,
      "last actual prepared owner failed to dispose/refund its known unstarted slot");
}
}
int main(int argc, char **argv) {
  try {
#ifdef MINIGPU_EXPECT_KNOWN_ALLOCATION_LAYOUT
    static_assert(mgpu::allocation_layout::sharedLayoutSupported, "Known setup witness needs the qualified Release compiler");
#endif
    if (!mgpu::allocation_layout::sharedLayoutSupported) {
      std::cout << "UNSUPPORTED setup allocation layout; no factory qualification\n"; return 0;
    }
    require(argc == 3 && std::strcmp(argv[1], "--layout-dll") == 0,
        "usage: minigpu_setup_layout_cpu --layout-dll ABSOLUTE_DLL_PATH");
#ifdef _WIN32
    const std::filesystem::path path(argv[2]); require(path.is_absolute(), "explicit absolute DLL required");
    struct Module { HMODULE value; ~Module() { if (value) FreeLibrary(value); } } module{LoadLibraryW(path.c_str())};
    require(module.value != nullptr, "actual original DLL load failed");
    using Query = int (*)(MGPUCaptureSetupLayoutV1 *, uint32_t);
    const auto address = GetProcAddress(module.value, "mgpuCaptureSourceGetSetupAllocationLayoutV1");
    Query query = nullptr; static_assert(sizeof(address) == sizeof(query));
    std::memcpy(&query, &address, sizeof(query)); require(query != nullptr, "original setup layout export missing");
    MGPUCaptureSetupLayoutV1 layout{};
    require(query(&layout, sizeof(layout)) == 0 && mgpu::allocation_layout::validateSetupLayout(layout),
        "original DLL setup table is incompatible/incomplete");
    factories(layout);
    std::cout << "PASS actual Root/Prepared setup factory requests, original DLL extents, ordinary/aligned dispatch and retained alias/free tails; native10="
        << layout.total_requested_bytes << " reporting-only, no GPU/engine/capture\n";
#else
    throw std::runtime_error("original native setup DLL witness unsupported on this platform");
#endif
  } catch (const std::exception &error) { probe::state.active = false; std::cerr << "FAIL " << error.what() << '\n'; return 1; }
}

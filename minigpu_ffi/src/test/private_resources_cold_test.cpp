// Actual new exported C/C++ refusal paths; no live lease, engine or GPU.
#include "../include/minigpu.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
extern "C" int private_resources_c_abi(void);
static void require(bool value, const char *why) {
  if (!value) {std::cerr << "FAIL " << why << '\n'; std::exit(1);}
}
int main() {
  MGPUContextLifetimeStats before{}, after{};
  const auto refs = mgpuContextRefCount();
  require(private_resources_c_abi() && mgpuContextGetLifetimeStats(&before, sizeof(before)) == 0, "actual C ABI/initial counts");
  auto *shader = reinterpret_cast<MGPUComputeShader *>(uintptr_t(1));
  auto *buffer = reinterpret_cast<MGPUBuffer *>(uintptr_t(1));
  auto *notALease = reinterpret_cast<MGPUContextLease *>(uintptr_t(1));
  require(mgpuContextLeaseCreateComputeShaderChecked(nullptr, &shader) == -1 && !shader,
    "missing lease clears shader without initialization");
  require(mgpuContextLeaseCreateBufferChecked(nullptr, reinterpret_cast<const MGPUPrivateBufferDescV1 *>(uintptr_t(1)), 40, &buffer) == -1 && !buffer,
    "missing lease refuses before descriptor access");
  require(mgpuContextLeaseCreateComputeShaderChecked(notALease, nullptr) == -1,
    "missing output does not borrow unknown lease");
  MGPUPrivateBufferDescV1 desc{40, 1, 4, 128, 0, nullptr, 0};
  desc.usage = 1; // MapRead cannot enter the narrow buffer profile.
  require(mgpuContextLeaseCreateBufferChecked(notALease, &desc, 40, &buffer) == -1 && !buffer,
    "unsupported CPU/map usage rejects before any lease access");
  desc.usage = 128; desc.byte_size = 128ull * 1024 * 1024 + 4;
  require(mgpuContextLeaseCreateBufferChecked(notALease, &desc, 40, &buffer) == -1 && !buffer,
    "oversized resource rejects before reservation");
  desc.byte_size = 4; desc.initial_data = &desc; desc.initial_size = 0;
  require(mgpuContextLeaseCreateBufferChecked(notALease, &desc, 40, &buffer) == -1 && !buffer,
    "initial pointer/length mismatch rejects before copy");
  MGPUPrivateShaderSetupV1 setup{40, 1, "x", 65537, 0, nullptr, 0, 0};
  require(mgpuComputeShaderSetupChecked(notALease, reinterpret_cast<MGPUComputeShader *>(uintptr_t(1)), &setup, 40) == -1,
    "oversized WGSL rejects before copy/owner access");
  setup.wgsl_size = 1; setup.reserved = 1;
  require(mgpuComputeShaderSetupChecked(notALease, reinterpret_cast<MGPUComputeShader *>(uintptr_t(1)), &setup, 40) == -1,
    "reserved fields reject before owner access");
  require(mgpuComputeShaderSetupChecked(nullptr, nullptr, nullptr, 0) == -1,
    "missing setup owners refuse");
  require(mgpuContextGetLifetimeStats(&after, sizeof(after)) == 0 && refs == mgpuContextRefCount() &&
    std::memcmp(&before, &after, sizeof(before)) == 0, "cold context accounting unchanged");
  std::cout << "PASS actual checked private-resource C/C++ ABI and cold owner/descriptor refusals; no GPU/engine/capture\n";
}

# Typed buffer ranges v1

Root-reviewed additive G01 contract. Keep all legacy binding APIs unchanged.
This exposes the existing compute backend's uniform/storage distinction with
explicit byte ranges; it is not a raster pipeline or WGSL reflection API.

## Public contract

`BufferBindingKind` has wire values `storage=0`, `readOnlyStorage=1`, `uniform=2`.
`BufferBindingDescriptor` is immutable: required `kind`, required `sizeBytes`,
`offsetBytes=0`. Both byte values fit 0..2147483644 and are multiples of four;
size is positive. Constructor throws ArgumentError for malformed values.

`ComputeShader.setBufferRange(int slot, Buffer buffer,
BufferBindingDescriptor descriptor)` is an ordered binding update. Slot is
0..31. Buffer and shader must be live. Descriptor-created buffers must have the
appropriate storage/uniform usage and the range must fit sizeBytes. Legacy
buffers remain usable but actual backend metadata remains authoritative.
Never infer binding kind from WGSL text or a property name.

PlatformComputeShader gets a concrete unsupported default of the same method
using PlatformBuffer. FFI/web implement it; old binaries explicitly refuse this
new method. There is no partial fallback that discards range/type information.

## C boundary and validation

```
uint32_t mgpuBufferBindingVersion(void); // 1
int mgpuSetBufferRange(MGPUComputeShader* shader, int slot, MGPUBuffer* buffer,
                       int kind, int offsetBytes, int sizeBytes);
```

Return 0 on accepted enqueue, 1 malformed/range/usage/foreign-context request,
2 unavailable device, 3 backend failure. Never let exceptions cross the C ABI.
Before enqueue validate pointers, range arithmetic without overflow, closed kind,
slot, actual buffer usage and exact buffer/shader MGPU owner identity. Query the
current device limits for maxBindingsPerBindGroup, matching offset alignment
and maxUniformBufferBindingSize/maxStorageBufferBindingSize. No new device or
adapter and no synchronous wait for a GPU submission.

The queued task snapshots raw buffer handle, size, offset and kind, follows the
existing bind/dispatch/destruction FIFO and updates the unified binding table.
Add read-only-storage to the native binding-kind switch. Changes of type, size
or offset must invalidate the pipeline layout/bind group even with the same
buffer handle. A later legacy setBuffer on the same slot must restore the
legacy full-storage binding; it cannot early-return based only on pointer.
Do not add asynchronous raw-buffer borrows outside the existing context queue.

Dart maps unavailable to StateError, invalid to ArgumentError and backend failure
to StateError; absence of the versioned API is UnsupportedError. Generic package
diagnostics contain no room identity or fixture-specific terms. Web keeps all
exports scoped under MinigpuModule.

## One coordinated acceptance

Extend the descriptor native scenario with one actual compute kernel consuming
uniform parameters at offset 256 and read-only storage at offset 256, writing
storage output. Read back exact values. Rebind the same handle with another valid
offset and exercise the legacy-binding transition. Reject unaligned/out-of-range,
wrong-usage and foreign-context inputs before submitting them to WebGPU. Run the
equivalent Dart/public scenario on native and browser where available. Root owns
the shared check; workers do not launch independent regression builds.

## Current evidence

Implemented September 9, 2026. Two public Dart boundary checks passed, covering
legacy byte-size fallback/refusal and typed range/usage/disposal behavior. The
dedicated native C ABI scenario passed with actual GPU allocation, compute and
readback, including an accepted same-context binding paired with the refused
foreign-context request. The standalone browser Dart/Wasm caller passed against
the newly built Emscripten backend in headless Chrome, with no uncaught runtime
errors. It exercised explicit allocation metadata, uniform and read-only storage
ranges, same-buffer offset changes and malformed-request refusal.

This is API/functionality evidence, not device-loss, mobile, rendering-performance
or full engine-fixture qualification. The browser build candidates are staged in
system temp during the larger graphics pass; the installed browser artifacts and
native application still need their normal build/release integration. Existing
compute callers and their packed element APIs are unchanged.

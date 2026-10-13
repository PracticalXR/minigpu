# Explicit buffer allocation v1

This additive API prepares ordinary vertex/index/uniform/storage resources for
the render backend. Existing `createBuffer(byteSize, dataType)` and its packed
type conversions retain their behavior. No new context or adapter is created.

`BufferDescriptor({required int sizeBytes, required int usage})` is immutable.
Size is an exact byte count, at least 4, a multiple of 4 and no greater than
2147483644 (the existing C ABI's signed allocation bound). Usage is a nonzero
mask of `BufferUsage.copySrc=4`, `copyDst=8`, `index=16`, `vertex=32`, `uniform=64`
and `storage=128`. Reject unknown bits. V1 does not expose mapping, indirect
commands or query resolve through this allocation route. `BufferUsage.legacy`
is copySrc | copyDst | storage. Invalid descriptors throw ArgumentError before
any backend call. Validation is repeated by native entrypoints.

`Minigpu.createBufferFromDescriptor(descriptor)` returns a normal owned Buffer,
tracks its lifetime exactly like legacy allocations, and preserves descriptor
metadata on the wrapper. Its native backing uses uint32 words (no packed-type
conversion). Prefer writeRawBytes for structured data. Byte descriptors do not
change existing element-count read/write APIs. Read/write methods reject absent
copySrc/copyDst usage on descriptor-created buffers. Unannotated legacy wrappers
retain existing behavior. Storage bindings require storage usage; uniform
bindings require uniform usage. This allocator does not add raster or general
binding-layout validation by itself.

The platform adds concrete default `createBufferFromDescriptor` which accepts
only `legacy` usage through the old uint32 allocation route, refusing other
usages as unsupported. FFI global/context and browser backends override it with
the explicit route; all constructor bounds are checked again at this boundary.
Older installed native/browser binaries may use the old route for legacy usage
only. They must refuse other usage, never pretend the bits were installed.

Additive C ABI:

```
uint32_t mgpuBufferDescriptorVersion();  // 1
MGPUBuffer* mgpuCreateBufferWithUsage(int sizeBytes, uint32_t usage);
MGPUBuffer* mgpuContextCreateBufferWithUsage(
    MGPUContextHandle* handle, int sizeBytes, uint32_t usage);
```

Invalid arguments or creation failure return null. A null context handle refuses.
Keep old functions/symbols unchanged. Both new creation functions use the owner's
existing device/worker, store explicit usage in BufferData, and preserve current
completion/destruction behavior. Do not rebuild generated JS/Wasm artifacts as
part of a source-only worker assignment; the primary owns one coordinated native
and browser verification path.

The WebGPU source definition makes buffer size a byte count and usage explicit:
[GPUBufferDescriptor](https://gpuweb.github.io/types/interfaces/GPUBufferDescriptor.html).
Bindings need separate size/layout validation; this descriptor is not a substitute
for [GPUBufferBindingLayout](https://gpuweb.github.io/types/interfaces/GPUBufferBindingLayout.html).

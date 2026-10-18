# Raster v1 implementation contract

September 9, 2026. Primary-reviewed G02/G03 contract, implemented for native and
browser. The receipts at the end record what has run. Revised October 7, 2026.
This is a generic graphics API, not a scene graph or guest-code permission API.
It builds on BUFFER_DESCRIPTOR_V1, BUFFER_BINDING_V1, TEXTURE_RESOURCE_V1 and
OWNED_RESOURCE_LIFETIME. Native and browser use the same C++ WebGPU encoder.

## Public API and limits

`Minigpu.createRenderPipeline(RenderPipelineDescriptor)` returns
`Future<GpuRenderPipeline>`. Descriptors are immutable, validate before backend
allocation, and own unmodifiable copies of collections. Preparation is
asynchronous; compilation must not block the Dart caller. The pipeline wrapper
is registered before awaiting preparation. Closure during preparation retires
it and rejects the creation future. A failed compilation reports a bounded
backend diagnostic, never a usable invalid pipeline.

Pipeline descriptor fields: `code`, `vertexEntry` (default `vsMain`),
`fragmentEntry` (default `fsMain`), `colorFormat`, `blend` (opaque by default),
`topology` (triangleList), `frontFace` (ccw), `cullMode` (none),
`depthEnabled` (false), `depthWrite` (false), `depthCompare` (lessEqual),
`vertexBuffers` (list of VertexBufferLayout), `bindings` (RenderBindingLayout).
Depth is depth32Float when enabled. Color formats are texture codes 1,2,3,6,7.
No format conversion, implicit UV flip, or gamma conversion outside the declared
texture format. Depth clip range 0..1, viewport origin top-left, sampleCount=1.

VertexBufferLayout has `arrayStride` bytes, `stepMode` vertex/instance and
`attributes`; VertexAttribute has `shaderLocation`, `offset` bytes and `format`.
At most 8 layouts, 16 total attributes, unique locations 0..31, strides 4..2048
and multiples of 4. Attributes fit their stride and meet format alignment.
Actual device limits apply in addition to these portable bounds.

Binding layouts use group 0, unique slots 0..31, vertex=1/fragment=2 visibility
bits (nonzero). RenderBindingKind: uniformBuffer=0, readOnlyStorageBuffer=1,
sampledFloat=2, sampledUnfilterableFloat=3, sampledDepth=4, filteringSampler=5,
nonFilteringSampler=6, comparisonSampler=7. Texture dimension uses existing
TextureViewDimension codes. Buffer layouts declare `minBindingSize` (positive,
multiple of 4, <=2147483644); other kinds use zero. Nontexture kinds require
dimension d2. Validate aggregate per-stage device binding limits. The actual
WebGPU pipeline compiler validates shader/layout compatibility.

RenderBlendMode wire codes: opaque=0; alpha=1 (RGB srcAlpha/oneMinusSrcAlpha,
alpha one/oneMinusSrcAlpha); premultipliedAlpha=2 (one/oneMinusSrcAlpha both);
additive=3 (one/one both). All operations add, all channels writable.
Topology triangleList=0,lineList=1,pointList=2; FrontFace ccw=0,cw=1;
CullMode none=0,front=1,back=2; VertexStepMode vertex=0,instance=1.
VertexFormat float32=0,float32x2=1,float32x3=2,float32x4=3,uint32=4,
uint32x2=5,uint32x4=6,sint32=7,unorm8x4=8. IndexFormat uint16=0,uint32=1.
Depth compare uses existing SamplerCompare wire codes 1..8.
Attribute offsets align to min(4, format byte size), not the whole vector size.
The blend type is deliberately distinct from Flutter's `BlendMode`.

`pipeline.createPass({required GpuTextureView color, GpuTextureView? depth,
List<double> clearColor = [0,0,0,0], double clearDepth=1,
bool loadColor=false, bool loadDepth=false})` creates a Dart command recorder.
Attachments must be alive, render-attachment usage, single-mip/layer d2 views,
same dimensions and the pipeline's exact format/depth requirements. Clear
components must be finite; depth 0..1. Both attachments are stored on completion.
Attachment textures cannot also be sampled within that pass (v1 conservative
whole-texture feedback refusal). No hidden device/surface creation.

`GpuRenderPass` methods: `setVertexBuffer(slot, Buffer, {offset=0,sizeBytes})`,
`setIndexBuffer(Buffer, IndexFormat, {offset=0,sizeBytes})`,
`setBuffer(binding, Buffer, {offset=0,sizeBytes})`,
`setTexture(binding, GpuTextureView)`, `setSampler(binding,GpuSampler)`,
`draw(vertexCount,{instanceCount=1,firstVertex=0,firstInstance=0})`,
`drawIndexed(indexCount,{instanceCount=1,firstIndex=0,baseVertex=0,firstInstance=0})`,
`setViewport(x,y,width,height,{minDepth=0,maxDepth=1})`,
`setScissorRect(x,y,width,height)`, `Future<void> submit()`, and `discard()`.
Sizes default to remaining bytes; explicit descriptors/usages are required.
Vertex/index usages and alignment are checked; uniform/storage ranges respect
device alignment, declared minimum size and actual device maximums. All required
bindings/vertex layouts must be set before a draw. Index data stays on GPU;
WebGPU robust indexed-access rules apply, not a CPU readback of index values.
Vertex/instance ranges, offsets and command arithmetic use checked bounds.
Positive counts only, <=16777216 vertices/indices and <=1048576 instances per
draw; first values uint32, baseVertex int32. Full attachment viewport/scissor
are defaults; explicit rectangles must stay within the target and be nonempty.

Each pass has one pipeline, <=4096 commands, <=1024 draws, <=65536 words,
<=256 referenced buffers/views/samplers per family. Zero draws permits a clear.
It pins public wrappers and rechecks liveness at submission. Crossing the backend
submission boundary consumes the pass, even if that backend subsequently refuses
it; local preflight failure does not consume it or submit a partial pass. Discard frees
host references. `submit` acknowledges host enqueue, NOT GPU completion;
`Minigpu.waitForSubmittedWork()` remains the completion fence. No command can
mutate or re-submit a consumed recorder. Native queued work owns its resources
independently until encoding completes: uploads and passes admitted before a
texture's destroy still run, and the destroy frees the GPU memory once they
have, even while a pipeline's cached bind group still names one of its views.
Submitted GPU use has WebGPU lifetime.

Context retirement closes pending/ready render pipelines before views/textures/
samplers/buffers and detaches afterward. Old-device pipelines and resources
refuse; explicit restoration builds a new pipeline. No automatic device reinit.

## C ABI (additive, versioned)

New `minigpu_render.h/.cpp`. Opaque `MGPURenderPipeline`. Every entry contains
exceptions. C pointers are trusted host wrapper pointers, never guest handles.

```
uint32_t mgpuRasterResourceVersion(); // 1
MGPURenderPipeline* mgpuCreateRenderPipeline(
  const char* wgsl, const char* vertexEntry, const char* fragmentEntry,
  const uint32_t* descriptor, int wordCount);
MGPURenderPipeline* mgpuContextCreateRenderPipeline(
  MGPUContextHandle*, const char*, const char*, const char*, const uint32_t*, int);
int mgpuRenderPipelineStatus(MGPURenderPipeline*); // 0 preparing, 1 ready, 2 failed, 3 closed/stale
// Posts (token << 1) | ok to the Dart port once, when the status leaves 0.
void mgpuRenderPipelineNotifySettledToPort(MGPURenderPipeline*, int64_t port, int64_t token);
int mgpuRenderPipelineDiagnostic(MGPURenderPipeline*, char* destination, int capacity);
double mgpuRenderPipelineBindingStat(MGPURenderPipeline*, int selector);
void mgpuDestroyRenderPipeline(MGPURenderPipeline*);
int mgpuSubmitRender(MGPURenderPipeline*, MGPUTextureView* color,
  MGPUTextureView* depth, const float* clearRgbaDepth, int loadFlags,
  MGPUBuffer* const* buffers, int bufferCount,
  MGPUTextureView* const* views, int viewCount,
  MGPUSampler* const* samplers, int samplerCount,
  const uint32_t* commands, int wordCount);
```

Null pipeline creation refuses descriptor/owner/unavailable/allocation failure.
WGSL is <=1MiB UTF-8 excluding NUL; entry identifiers ASCII identifier syntax,
1..64 bytes. Copy strings/descriptors before returning. Use asynchronous WebGPU
pipeline creation, shared backing state, retained creation device and a callback
that never touches a destroyed owner. Native worker may pump compilation events;
browser must yield naturally, never block/spin the JS event loop. During cold
preparation the Dart adapter waits for the settle signal: the native port post
above, or on the web the module calling `Module.mgpuRenderPipelineSettled(handle)`
once per pipeline. A backend without the signal is polled from 1ms, backing off
to 16ms. Either way it must stop on wrapper closure and cache failure, with
bounded 30-second caller wait.
Timeout destroys that pipeline and prevents late completion from resurrecting it.
Diagnostic copies at most min(capacity-1,512) UTF-8 bytes, NUL terminates when
capacity>0, returns copied length. No pointer to internal string storage escapes.

Descriptor is uint32 words, exact consumption required, max4096 words:
header `[1,colorFormat,blend,topology,frontFace,cullMode,depthEnabled,depthWrite,
depthCompare,vertexBufferCount,bindingCount]`;
each vertex layout `[stride,stepMode,attributeCount]`, then each attribute
`[location,offset,format]`; each binding `[slot,kind,visibility,viewDimension,
minBindingSize]`. Flags are 0/1. Inactive depth requires depthWrite=false.

Commands are concatenated fixed records, no padding, exact consumption:

| Opcode | Words including opcode |
| --- | --- |
| 1 vertex buffer | `[1,slot,bufferIndex,offset,size]` |
| 2 index buffer | `[2,bufferIndex,offset,size,indexFormat]` |
| 3 shader buffer | `[3,bindingSlot,bufferIndex,offset,size]` |
| 4 shader texture | `[4,bindingSlot,viewIndex]` |
| 5 shader sampler | `[5,bindingSlot,samplerIndex]` |
| 6 draw | `[6,vertexCount,instanceCount,firstVertex,firstInstance]` |
| 7 indexed draw | `[7,indexCount,instanceCount,firstIndex,baseVertexBits,firstInstance]` |
| 8 viewport | `[8,xBits,yBits,widthBits,heightBits,minDepthBits,maxDepthBits]` |
| 9 scissor | `[9,x,y,width,height]` |

Float fields use IEEE754 float32 bit patterns. `loadFlags`: bit0=color load,
bit1=depth load, no other bits; absent depth cannot load depth. Clear float array
is exactly RGBA then depth. Pointer arrays point to existing live wrappers for
the duration of admission; copy commands and retain all backing resources before
returning. Native BufferData must retain its creation device across moves until
release, so same-owner handles from an old device can be refused.

Submit status follows texture ABI: 0 accepted enqueue,1 invalid descriptor/range/
usage/owner,2 stale/unavailable,3 backend failure. Validate the ENTIRE pass before
any clear/draw; submission task rechecks device and owned texture/pipeline
liveness, flushes prior compute work, encodes one render pass and submits. No
cross-device copy or software-renderer fallback. Unknown opcodes refuse.

## Retained bind groups and lifecycle counters

Added September 14, 2026; described here October 7, 2026. Neither changes what a
pass draws.

Each render pipeline retains the bind groups its draws have used, so a stable
binding is not rebuilt for every pass. An entry is keyed by the exact binding
set of one draw: for every layout slot, the buffer with its offset and size, or
the texture view, or the sampler. Any difference is a different entry. A
pipeline holds at most 64 entries. When a draw needs a 65th, the entry least
recently used by a draw is released first, and an evicted binding set is built
again on its next use. An entry keeps its buffer, view and sampler handles
referenced until it is evicted or the pipeline is destroyed, so a handle reused
for a new resource can never match an old entry, and a bind group for a
resource its owner has destroyed is never matched again. Eviction releases only
the cache's reference: a pass that is being encoded keeps the bind groups it
has already set. The cache is consulted once per draw whose bindings changed
since the previous draw of the same pass; a pipeline without bindings never
uses it.

`mgpuRenderPipelineBindingStat(pipeline, selector)` returns monotonic counters
for one pipeline: selector 0 bind groups created, 1 reused, 2 retired, 3 live
(created minus retired). A null pipeline or an unknown selector returns zero.
The value is a double so that native and browser callers receive the same
number type; counts are exact through 2^53.
`GpuRenderPipeline.bindingCacheSnapshot` exposes the four values. It is null
when the loaded backend predates the export and after the pipeline is
destroyed: a destroyed pipeline reports no counters, not zeros.

`Minigpu.resourceAllocationSnapshot` reports created, retired and live counts
for the buffers, textures, texture views, samplers, compute shaders and render
pipelines of one wrapper. Created and retired never reset when the device is
reopened, so a create-and-retire cycle stays visible when every live count has
returned to its earlier value. A render pipeline is counted when its creation
is requested, before preparation ends, and is retired when it is closed, fails,
times out or its context is retired. These counters describe wrapper
allocations. They are not driver memory, a memory-pressure signal or a
performance measurement.

## Assigned evidence

One native scenario and the same public Dart/Wasm browser scenario: indexed
instanced geometry with explicit vertex/color data, depth occlusion and alpha
blend into an owned texture; sample/readback a few exact/toleranced pixels with
the existing compute API. Include malformed descriptor/command, insufficient
buffer usage/range, wrong owner, destroyed attachment and close-during-prepare
cases at the applicable public/native boundaries. Root owns checks. This does
not qualify mobile devices, actual presentation, performance or an engine graph.

## Implementation receipt, September 9, 2026

Implemented the public Dart recorder/descriptors, native C ABI, default and
secondary-context FFI adapters, and browser Wasm adapter. Root reviewed actual
changes and corrected command length, destruction/callback races, sampler
compatibility, input bounds and lifecycle handling before qualification.

- `minigpu_render_resource_test` built and passed on Windows Dawn D3D11,
  NVIDIA RTX 3090: indexed instancing, depth and alpha blend into an owned
  texture, compute/readback, malformed/usage/owner/stale refusals and pending
  pipeline destruction. Deliberate context destruction logged device loss.
- `minigpu/test/raster_browser_smoke.dart` compiled to Dart/Wasm and passed
  in isolated headless Chrome against a temporary backend candidate, with no
  captured browser errors: vertex/instance/index/uniform buffers, sampled
  texture/sampler, viewport/scissor, depth occlusion, load/store and alpha blend,
  pixel checks, missing/range/stale refusal and context retirement.
- Targeted Dart analysis found no errors. Existing adapter warnings/lints and
  descriptor brace-style notices remain; this is not a clean whole-package
  analysis claim.

The native receipt exercises the exported C ABI; the browser receipt exercises
the public Dart API and Wasm marshaling. They do not qualify the native Dart FFI
route on every platform. On September 9 the build candidates were in temp/cached
build output and the packaged browser assets had not been replaced. The packaged
pair has since been rebuilt from this source into the package's normal web
assets, and the October 7 browser receipt below ran against it; nothing here is
deployed. Guest graph admission, renderer integration and the platform/device
matrix remain separate.

## Route receipt, September 10, 2026

The native public Dart API route was verified through the shared
`minigpu/minigpu/test/raster_scenario.dart`, extracted from the existing browser
scenario and reused by `native_smoke`. One selected native wrapper run against
cached native assets passed on Windows Dawn D3D11 with an NVIDIA RTX 3090: indexed
and instanced raster with depth, alpha blending, texture/sampler/uniform use,
compute pixel readback, missing/range/stale attachment refusal and owned-context
retirement. The only device-lost message was the deliberate context destroy.
The existing browser receipt was reused; its wrapper was not rerun until the
October 7 receipt below. Mobile and device-performance qualification remain
open. Presentation and an engine graph are outside this contract's evidence;
the layers that own them keep their own receipts.

## Receipts, October 7, 2026

Native runs: Windows, NVIDIA RTX 3090, a fresh Release build of this source
linked to the prebuilt Dawn. D3D11 is the default on Windows whatever the
adapter topology; `MGPU_BACKEND=d3d12` selects D3D12. Each run's log names its
adapter and backend (`backendType=3` is D3D11, `backendType=4` is D3D12).

- `minigpu_render_resource_test` (C ABI, unchanged) passed on D3D12 and on
  D3D11, each with an empty and with a warm shader cache: indexed instancing,
  depth and alpha blend, readback, bind-group reuse, malformed, usage and stale
  refusals, destruction of a pending pipeline and refusal of a second
  context's attachment.
- `minigpu/test/raster_native_smoke.dart` (the shared `raster_scenario.dart`,
  unchanged) passed through the public Dart API on D3D12 and on D3D11.
- `minigpu/test/raster_lifecycle_native_smoke.dart` (new; it runs the shared
  `raster_lifecycle_scenario.dart`) passed three times on each backend:
  - Bind-group cache, as created/reused/retired/live: 64 distinct binding sets
    give 64/0/0/64 and 64/64/0/64 when drawn again. After the oldest entry is
    used once more, a 65th set evicts the least recently used entry and not
    that oldest one (65/65/1/64; the oldest is then still a reuse, 65/66/1/64).
    The evicted set is built again on its next use and draws its own colour
    (66/66/2/64). 128 sets in one pass end at 192/68/128/64 with every pixel
    correct. Buffers retired while their groups were cached, then replaced,
    end at 248/68/184/64, and the replacements are reused (248/132/184/64).
  - Closure during preparation: eight pipelines were closed while the backend
    still reported each one preparing. Every preparation rejected and none
    became usable when its compilation finished. Context retirement rejected
    eight pending creations and retired their wrappers at once.
  - Wrong owner: a pass refused an attachment, and a buffer, created by a
    second context on the same adapter, at the public Dart wrapper.
- Each of these claims was seen to fail against a mutated copy of the source:
  no cache bound, no recency on reuse, an evicted group handed to the new key
  (caught by the pixels alone), no attachment owner check, no buffer owner
  check, a closure that does not reject, and a context retirement that skips
  its pipelines.

Browser runs: the installed desktop Chrome 155.0.8059.40 on the same
workstation, headless, with no WebGPU flag, a hardware NVIDIA "ampere" adapter
(the RTX 3090), a cross-origin isolated page, and entries compiled with
`dart compile wasm`. The backend was the pair packaged in `minigpu_web/lib/web`,
served unchanged. There was no page error, console error or uncaptured GPU
error, and the only device losses were the deliberate destroys.

- `minigpu/test/raster_browser_smoke.dart` (unchanged) passed.
- `minigpu/test/raster_lifecycle_browser_smoke.dart` (new) passed twice:
  - Closure during preparation: eight of eight pipelines were preparing when
    closed; the results are those of the native run.
  - Bounded preparation: the page withheld the delivery of two real
    `createRenderPipelineAsync` results. Both preparations were still pending
    at 1, 15 and 29 seconds while a 100 ms timer kept firing, rejected with a
    timeout after 30.0 seconds and destroyed their pipelines. Delivering the
    two results afterwards revived nothing and changed no counter, and the
    same two sources then prepared and drew normally.
  - The bind-group cache recorded the native counters at every stage.
- Mutated copies failed here as well: a wait that gives up without destroying,
  a 300-second bound, and an adapter that forgets a closure when the backend
  completes late.

Not qualified by these receipts: mobile, other desktop browsers, other native
platforms, device performance, a deployed artifact, and a compilation that is
slow by itself (the late completions were withheld, not slow). On D3D12 a
separate probe with no raster in it crashed when a buffer was destroyed and
another created straight away. The lifecycle check lets retirements finish
before it allocates, and that defect belongs to the buffer lane, not to this
contract.

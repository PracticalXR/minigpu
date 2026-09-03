# Minigpu FFI

This is the FFI implementation of the minigpu package.

see https://pub.dev/packages/minigpu

## Dawn build location

minigpu depends on [Dawn](https://dawn.googlesource.com/dawn) (Google's WebGPU
implementation). On first build Dawn is cloned and compiled automatically.

| Platform | Default path                  |
|----------|-------------------------------|
| Windows  | `%SYSTEMDRIVE%\dawn` (e.g. `C:\dawn`) |
| macOS    | `~/dawn`                      |
| Linux    | `~/dawn`                      |

Windows uses a short root-level path to stay well within the 260-character
`MAX_PATH` limit that Dawn's deeply-nested source tree can otherwise exceed.

### Override: `MINIGPU_DAWN_DIR`

Set the `MINIGPU_DAWN_DIR` environment variable to point Dawn at a different
root directory. The build expects a `build_{os}_{arch}/` subdirectory inside
it (e.g. `build_win_x86_64/`).

```powershell
# Windows example — use a custom drive/path
$env:MINIGPU_DAWN_DIR = 'D:\my_dawn'
```

```bash
# macOS / Linux example
export MINIGPU_DAWN_DIR=/opt/dawn
```

This env var is read by both the Dart build hook (`hook/build.dart`) and the
CMake module (`src/cmake/dawn.cmake`), so it works whether you build via
`flutter build`, `dart pub get`, or `cmake` directly.

Set it **before** the build; the hook resolves it into the `-DDAWN_DIR=…` it
passes to cmake, and that define outranks everything else. Setting `DAWN_DIR` in
your app's `windows/CMakeLists.txt` has no effect — minigpu's native asset is
built as its own cmake project, not a subdirectory of your app.

> Fixed in 1.5.8: before that, `MINIGPU_DAWN_DIR` changed only where the hook
> looked for a prebuilt `webgpu_dawn` library. cmake was still told the platform
> default (`%SYSTEMDRIVE%\dawn`) and would clone and build Dawn there, so the
> variable appeared to be ignored.

### Troubleshooting

**`Invalid character escape '\d'` / `Syntax error in cmake code at
<dawn>/build_win_x86_64/tmp/CMakeLists.txt`**

Fixed in 1.5.8. A Windows Dawn root reached `FetchContent_Declare` with
backslashes; FetchContent copies `SOURCE_DIR`/`BINARY_DIR` verbatim into a
sub-build `CMakeLists.txt` it generates, and there `C:\dawn` is a string with an
invalid escape. `dawn.cmake` now normalizes `DAWN_DIR` with
`file(TO_CMAKE_PATH …)`. Only machines **without** a prebuilt Dawn hit this — when
an existing build is found, the from-source branch never runs. On an older
version, pass a forward-slash path (`-DDAWN_DIR=C:/dawn`) or install a prebuilt
Dawn at the default path.

**`webgpu_dawn shared library not found`**

The root you pointed at must contain `build_{os}_{arch}/` — e.g.
`E:\my_dawn\build_win_x86_64\`, with the import library at
`build_win_x86_64/src/dawn/native/Release/webgpu_dawn.lib` and the DLL somewhere
under that tree. A Dawn built with a different output layout (`out/Release`, a
bare `build/`) will not be found.

**`em++: error: … not a valid port path: …emdawnwebgpu-v<stamp>.remoteport.py`**

Fixed in 1.5.8. Dawn ships the Emscripten port file as either the in-tree
`src/emdawnwebgpu/pkg/emdawnwebgpu.port.py` or a generated, version-stamped
`emdawnwebgpu-v<stamp>.remoteport.py`. The web build hardcoded the stamped name,
so bumping `DAWN_COMMIT` broke it — and em++ only complained ~370 targets in,
after Dawn itself had compiled. It is now detected at configure time (look for
`-- emdawnwebgpu port -> …` in the log) and a miss is a `FATAL_ERROR` naming
`DAWN_COMMIT`.

## Shader compilation: the persistent blob cache

Dawn compiles WGSL to a backend shader at pipeline creation and caches the
result — but only in memory, unless the embedder gives it somewhere durable to
write. minigpu now does, so compilation is a once-per-machine cost instead of
once-per-process. Nothing to enable; it is on by default.

This matters most on the **D3D11 backend**, which cannot consume DXIL, so every
shader goes through FXC, whose optimiser cost grows superlinearly with the body
size. A large compute kernel can sit there for tens of seconds on every launch.

Where the blobs go, and how to control it, is documented in the `minigpu`
package README. The parts specific to this layer:

- **One INFO line per device creation**, and nothing louder. Per-entry hit/miss
  detail is DEBUG only — this log channel defaults to INFO and a line per blob
  would bury real warnings, the same mistake per-buffer logging made once:

  ```text
  [mgpu shadercache] C:\Users\you\AppData\Local\minigpu\shadercache\v1 entries=64 bytes=2311044 cap=256MB hits=64 misses=0 stores=0 evictions=0
  ```

  This line is also how you confirm a rebuilt binary actually loaded: it names
  the directory and the counters. Check it rather than a DLL timestamp.

- **`MGPU_SHADER_CACHE=0`** disables caching, and **`MGPU_SHADER_CACHE_DIR`**
  redirects it. Both outrank the programmatic setters, because the point of an
  env var is to change a binary you cannot edit — the same precedence
  `MGPU_ADAPTER_NAME` has over `mgpuPreferDisplayAdapter`. Reach for
  `MGPU_SHADER_CACHE=0` first when a shader misbehaves and you want to know
  whether the cache is involved.

- **A failure is always a miss, never a failed device.** Unwritable directory,
  full disk, corrupt or truncated entry, antivirus lock, a race with another
  process — each costs a recompile and nothing else. `mgpuCreateContext` cannot
  fail because of this feature.

- **Stale blobs are prevented, not tolerated.** The key folds in the Dawn
  version, adapter, driver version and compile options, so a driver update or a
  GPU swap misses. Each entry additionally stores its own full key and is
  checked on read: the filename is only a hash, and Dawn's own hash validation
  binds hash→value rather than value→key, so without that check a filename
  collision could hand back a blob belonging to a different pipeline.

- **Writes are atomic** (temp file in the same directory, flushed, renamed) and
  readers open with full sharing, so eviction can never fault an in-flight read
  and no reader sees a half-written blob.

- **`mgpuShaderCacheSetProvider` is C-only and deliberately not exposed to
  Dart.** Dawn's load callback is synchronous and may arrive on a Dawn-internal
  thread; a Dart isolate can only be entered asynchronously from a foreign
  thread (`NativeCallable.listener`), which cannot return a blob to a blocked
  native caller. A Dart provider deadlocks. If you need custom storage from
  Dart, pre-populate the directory before init.

Counters are queryable (`mgpuGetShaderCacheStats`, or `Minigpu.shaderCacheStats`
from Dart). `pipelineCreateMs` is the total time spent in compute pipeline
creation — compare a cold launch against a warm one; that ratio, not any single
absolute number, is the signal.

## Buffer readback / upload performance

`FfiBuffer` pools its native scratch memory across frames to eliminate per-call
`malloc`/`free`, and completions cost no allocation at all.

### `writeRawBytes` is copy-free

The raw-upload path (`writeRawBytes`) does **not** stage through a scratch at
all. `TypedData.address` in a leaf call gives C the caller's own backing store,
so the payload is copied exactly once — by `wgpuQueueWriteBuffer`, into the
driver's staging ring. It used to memcpy the whole payload into a native scratch
first, purely to obtain a stable address; on a streaming path that was a
whole-frame host copy per frame, measured at 4K as **half the entire upload
stage** (3.71 → 1.84 ms/frame, with the time inside `wgpuQueueWriteBuffer`
unchanged).

Payloads over 32 MiB are still chunked, so the driver's staging ring never has
to hold more than one chunk; the chunks are `Uint8List.sublistView`s, which
alias rather than copy.

The remaining copy is the driver's own. Removing it too would require writing
the frame directly into a persistently mapped upload buffer, which is an API
change (the caller would have to assemble its frame in memory minigpu owns)
rather than something this path can do transparently.

### Read staging grows, it does not resize

The read path keeps one staging buffer per `Buffer`, and its capacity **only
grows** (to the next power of two, bounded by the source buffer's size). It used
to be destroyed and recreated whenever the next read asked for a different
length, which is invisible for a fixed-size tensor and pathological for a
payload whose length changes every frame: measured over 723 reads at 4K, the
destroy/create cycle cost 33.05 ms against 0.28 ms for grow-on-demand.

**How the pooled scratch works:**

- The first `read()` or `write()` call allocates a scratch buffer of the
  required size, kept alive on the `FfiBuffer` instance.
- Subsequent calls of the same size reuse it, so the hot path has zero FFI
  allocation overhead.
- If the required size grows the scratch is reallocated; if a read or write is
  already in flight (re-entrant across an `await`) a local one-shot buffer is
  used as a fallback — this is safe because Dart is single-threaded.
- `destroy()` frees the pooled scratch exactly once, guarded by a `_destroyed`
  flag. It deliberately tears down **no callback state**: see below.
- Completion of the async read is an int64 posted to a Dart port, so there is
  no per-call `NativeCallable` to register or close. That is a correctness
  requirement before it is a performance one — closing a callback the C layer
  may still invoke aborts the process — and it is documented in full in
  `lib/minigpu_ffi_completion.dart`.

## Batched transfers (scoped uploads / readbacks)

Per-range `writeBuffer` / `readSync` each take the device mutex and force their
own submit. When a frame issues many scattered transfers, that per-call tax
dominates. Two **additive** scopes collapse N transfers into one submit:

```c
mgpuBeginUploads(ctx);
mgpuStageWrite(buf, dstOffset, src, len);   // xN — memcpy into a host arena
mgpuEndUploads();                            // 1 queue write + N copyBufferToBuffer

mgpuBeginReadbacks(ctx);
mgpuStageRead(buf, srcOffset, dst, len);    // xN — records copies only
mgpuEndReadbacks();                          // 1 submit + 1 fence + 1 map
```

Both emit their copies into the batch encoder already open, so no extra submit
is forced. Scopeless or unaligned ranges fall back to the plain inline call, so
a caller can route every transfer through these unconditionally.

**Contract (read this — reads differ from writes):**

- `End` runs **inline on the caller's thread**, under the same device mutex the
  inline calls take. Ordering is identical to `writeBytesAt` / `readSync`; what
  is removed is the submit. Nothing moves onto the WebGPU worker thread —
  inline `mgpuReadSync*` / `mgpuWrite*` do **not** join that thread's FIFO, so
  emitting there would be a race.
- A staged **read is not filled until `End`**. Keep `dst` alive and do not
  inspect it early. Reads observe their source **as of `End`**.
- A scope must **not** span a dispatch that reads what it stages.
- `mgpuStageRead` takes **raw bytes** (mirror of `mgpuWriteBufferAt`), not the
  element counts `mgpuReadSyncUint32` uses.

Profiling: `mgpuUploadStats()` / `mgpuReadbackStats()`, or set
`MGPU_UPLOAD_PROF=1` / `MGPU_READBACK_PROF=1`.

**Measured (RTX 4090):** uploads removed the per-call tax (0.35–0.81 → 0.092 ms
on a ~97-write bracket) but did **not** move the wall — that path is byte-bound.
Readbacks *did*: 8 reads of 2.72 MB, 1.098 → 0.465 ms of API time. Batched
readback now sits at the hardware floor (submit 0.09 / map 0.24 at PCIe /
memcpy-out 0.13 at host speed).

## Ordering: binds, dispatches and destruction

Three things run on the WebGPU thread's FIFO and are therefore ordered against
each other: **buffer binds**, **dispatches**, and **shader destruction**.

- `mgpuSetBufferFire` / `ComputeShader::setBufferQueued` enqueue the bind. As of
  minigpu 1.5.9 the Dart facade routes *every* bind through this, so
  `setBuffer` / `setBufferAtSlot` are correct with `dispatchFire` as well as
  `dispatch`. Before that the binds ran inline and racing was silent: with
  `dispatchFire`, every fired dispatch saw the LAST binding.
- `mgpuDestroyComputeShader` queues the delete rather than freeing inline,
  because a queued bind captures the shader pointer and mutates its binding
  tables when it runs.
- `mgpuDestroyBuffer` still frees inline, and that asymmetry is deliberate: a
  queued bind captures only the raw `WGPUBuffer` handle **by value**, so it never
  touches the `Buffer` object.

What is **not** on that FIFO: `mgpuReadSync*` and `mgpuWrite*` run inline on the
caller's thread. So `mgpuDispatch` followed by `mgpuReadSync` is racy for any
caller — the read can overtake queued work. Use an awaited read (which flushes)
as the synchronization point.

Video textures are the remaining gap: `mgpuSetVideoTexture` binds inline and
`mgpuDestroyVideoTexture` frees inline, so a texture-bound shader must use an
**awaited** `dispatch`, never `dispatchFire`.

## GPU waits and the Windows timer quantum

`drain_dawn_events_with_timeout` previously waited with a 1 ms timeout on a
future that nothing could notify early, and a 1 ms condition-variable wait on
Windows rounds up to the system timer granularity (~15.6 ms) — so **every GPU
wait cost one whole tick**. It now probes with a zero timeout and yield-spins
for a bounded budget before degrading to coarse sleeping.

- `MGPU_DRAIN_SPIN_MS=<ms>` tunes the budget (default 8); `0` restores the
  pre-fix behaviour exactly, for A/B.
- `mgpuDrainSpinBudgetMs()` reports what the **loaded binary** implements — use
  it to assert you are not running a stale artifact.

Affects every caller that waits on GPU work through this drain: the shared
output texture present, Dawn-side debug reads, and the video import path.
Measured through a zero-readback present: p50 **15.69 → 2.57 ms** at 720p and
**15.69 → 10.07 ms** at 4K. (The pre-fix number being *identical* at 9× the
pixels is the tell — a cost invariant in the work is a clock tick, not work.)

## Attributing a frame-time spike: `MGPU_WAIT_PROF`

Transfer-path spikes are hard to attribute from a consumer's own stage timers,
because **the cost does not stay where it is spent**. `wgpuQueueWriteBuffer` is
asynchronous, so its device-side execution is charged to whichever call next
waits on the device. The visible symptom is a spike that moves between stages
run to run — it lands on `upload` once, on `readback` the next time — with no
correlation to the data. Do not read that as several different bugs: it is one
sync point absorbing whatever the device still owed.

`MGPU_WAIT_PROF=1` reports, per site, count / total / mean / max **and the spin
iteration count**, which is the discriminator those stage timers cannot give
you:

- a long wait with **thousands of spin iterations** is the device genuinely
  busy — the time is real GPU work and no host-side change will remove it;
- a long wall against a **handful of iterations** is the OS descheduling the
  thread, i.e. a scheduling or timer-granularity artifact.

`MGPU_WAIT_PROF_MS=<n>` additionally logs every individual event over n ms as it
happens, with a timestamp, so an outlier can be lined up against the consumer's
own per-frame output instead of being averaged away.

Sites: the read path's `lock` / `flush` / staging `alloc` / `submit` / `map`
wait / `copy` out / trailing event pump, the write path's `lock` / `flush` /
`write`, `q.lat` (the worker thread's enqueue-to-start latency) and `drain`.
Off by default, behind a single cached bool.

## Shared-texture present: correctness

`mgpuCopyBufferToSharedOutputTexture` returns only after
`wgpuQueueOnSubmittedWorkDone`. That wait is **load-bearing**: removing it
(`MGPU_PRESENT_NO_WAIT=1`, provided for testing) measured **11 torn + 28 stale**
consumer-visible surfaces out of 40 at 4K. The async twin still performs the
same wait, just on the worker thread, so it is not a fire-and-forget shortcut.

`mgpuDebugConsumerChecksumSharedHandle(handle, w, h)` hashes the surface through
an **independent** `ID3D11Device` — the consumer's view. The older
`…DebugReadFirstPixel` reads through Dawn's own device, whose immediate context
serialises against Dawn's submissions, making it *structurally incapable* of
observing a torn present; use the consumer checksum for any tearing gate.

> **Known gap:** `mgpuCopyBufferF32ToSharedOutputTexture` and
> `mgpuVideoTextureBGRAToRGBASharedOutput` return **without** that completion
> wait. Measurement says single-device D3D11 ordering holds at 720p (60/60
> fresh) but **breaks at 4K**, where the copy is large enough to lose the race.
> They have no live Dart callers today; fix them before adopting either.

# Owned resource lifetime: buffer and compute lane

G01 host-wrapper contract, September 9, 2026. New texture and raster resources
must build on a context that closes its children before detaching its backend.
The existing process-global attachment count and secondary-context policy stay
unchanged. No extra device, global reset or cross-isolate teardown is introduced.

## Fixed behavior

`Minigpu` tracks its buffers with weak references, preserving finalizer cleanup
for otherwise unreachable buffers. Explicit destruction and finalization remove
the registration and decrement the existing count exactly once. Do not retain
every buffer strongly until context shutdown. Reuse the existing live-shader
list; do not add a second shader cache or tracker.

Context destruction fences new creation immediately, retires all live shaders
and buffers before platform detach, and clears the copy shader and rebind
sentinel. Iterate snapshots because child destruction unregisters children.
Concurrent destroy calls share a single retirement future. If initialization is
in flight, destruction waits for that attempt and then closes its resources;
no late successful init can resurrect a closing wrapper. Init during destruction
waits for retirement before beginning a fresh attachment. Old child wrappers stay
destroyed after a successful new init.

All compute methods that touch the backend reject a destroyed shader, and buffer
binding rejects a destroyed buffer before dereferencing its platform pointer.
`destroy()` remains idempotent at the resource level. Counters are derived from
actual registrations: repeated CachedComputeShader.destroy cannot decrement a
different live shader's count.

Keep the historical `destroySync()` signature for source compatibility. It
begins the same fenced teardown; on asynchronous platforms its name cannot
promise completion. Attach an error handler so this legacy fire-and-forget call
cannot produce an unhandled rejection. Callers requiring a completion/error
receipt must await `destroy()`. Document that distinction, including an init
immediately following destroySync waiting for the pending teardown.

Cleanup failures must not skip other children or platform detach. After attempting
all retirements, awaited destroy reports a failure and leaves the wrapper closed.
Do not turn a failed cleanup into a successful "all resources released" claim.
The failed retirement receipt remains cached: subsequent init and destroy calls
report that failure. An unknown failed cleanup requires a new host/context;
there is no automatic reset that could reuse a partially retired attachment.
No changes to allocation byte units, packing, GPU completion semantics, room
grants or background budgets belong to this lane.

## Scope and evidence owner

Root owns one targeted lifecycle check against a recording platform: close while
initialization is pending, close with retained buffer/shader wrappers, and reopen
with old wrappers fenced. Record child-before-context ordering and exact counts.
The actual native/browser buffer scenario already qualifies the backend resource
path; do not independently rerun it for a Dart-only lifetime change.

This lane does not yet cover imported-video/shared-output wrappers, device-loss
generation reporting or arbitrary handles passed by a native embedder. Those
remain explicit G01/M02 work; wrappers do not certify untrusted C pointers.

## Evidence

Five focused owning-wrapper checks passed: close during pending init; retained
children retiring once before detach and remaining fenced on reopen; attempted
cleanup of every child and context after failure, with reopening refused; and
retry after synchronous initialization failure; and texture/view/sampler cleanup
before context detach with once-only retirement after a late finalizer callback.
Weak registrations retain retirement tokens so a cleared weak target cannot
escape context cleanup while its finalizer is pending. Targeted analysis was clean.
The recording asynchronous platform tests ownership/lifecycle, not GPU execution;
the separate actual native/browser buffer receipt supplies that evidence.

`Minigpu.withPlatform` lets an embedder supply an independent adapter wrapper
without modifying the default singleton. Construction creates no device; the
same explicit init/attachment/retirement contract applies.

## Native creation-device retirement

Native context loss no longer permits implicit replacement through buffer,
device, queue or instance access. The old creation handles remain old handles;
new buffer allocation and explicit initialization refuse until actual consumer
retirement permits context teardown. Native async attachment results track the
attachment itself, including a loss racing callback delivery. A failed result
must not be ignored by the Dart platform adapter.

The trusted C ABI `mgpuContextLeaseRetireDeviceAsync[ToPort]` requires a lease
retained while the device was ready and producer fencing by the caller. It
invalidates readiness, then destroys that exact creation device on its GPU
worker. Native Dawn destruction returning is evidence that its GPU timeline has
ended (possibly aborted), not that submitted rendering succeeded. External/CPU
readers, exports, process workers and leases remain independently owned. The
operation does not release them or authorize a context reset. Failed destruction
keeps the creation pin charged in bounded quarantine; pending work has no
timeout-to-success. Repeated calls cannot target a different generation.

This native destruction guarantee is not exported to Emscripten: the operation
refuses without altering readiness. A browser loss promise alone is insufficient
for this native guarantee. A host recovery implementation must keep actual GPU
retirement distinct from successful frames and from external-consumer cleanup.
The wrapper coordinator continues to retain all unresolved owners.

Native `context_lifetime_test --retirement` covers the real device path, retained
consumer charges, refusal to resurrect, and explicit replacement after consumer
retirement. The Dart native test covers propagation of initialization refusal.
These are deliberate destruction checks, not physical driver-loss qualification.

## Typed GPU resource retirement

`Minigpu.retireSubmittedWork(deviceEpoch: ...)` separates `completed` from
`deviceAborted`. Both can certify the end of GPU access for resource cleanup;
only successful `waitForSubmittedWork()` can certify frame completion. A healthy
queue error never authorizes device destruction. The creation epoch is checked
before and after the receipt, and a failed abortion receipt remains cached.

The native FFI adapter retains its exact creation-device lease while ready.
It refuses healthy/wrong-generation abortion and does not reacquire a lease on
loss. Releasing the adapter's pin does not certify final native context teardown:
the final native attachment retires asynchronously and other consumers/exported
resources may still own pins. External readers and host texture registration
must be joined separately; their failure remains a coordinator quarantine.
The base platform declines lost-device retirement. The web adapter now provides
the distinct GPU-only receipt below. Host recovery remains restricted to confirmed
retirement; there is no quarantine reset.

## Browser GPU-only retirement

The web adapter retains its exact creation `GPUDevice` and that device's `lost`
promise after fencing readiness. Only a fulfilled loss promise authorizes
`retireLostDevice`; a missing/replaced module handle, unavailable limits or
rejected promise cannot authorize destruction. The operation invokes idempotent
`destroy()` on the retained device, joins its loss promise and rechecks its
generation. Healthy/stale calls refuse and repeated valid calls share a receipt.
No ambient global or newly looked-up device is destroyed. Direct initialization
also refuses while the previous creation device remains attached.

This implements the WebGPU [device-loss and destruction contract](https://gpuweb.github.io/gpuweb/#dom-gpudevice-destroy)
for **GPU resource access only**. It cannot attest successful frame output,
external-reader/worker cleanup or final module/context destruction. The existing
coordinator continues waiting for those owners independently. Final browser
context observation is a separate receipt, described below. Browser adapter
names are not physical identity evidence.

The opt-in public test `minigpu/test/device_retirement_browser_smoke.dart` passes
against release Dart/Wasm and actual Chrome WebGPU: healthy/stale refusal,
unavailable lookup without device destruction, deliberate device loss, GPU buffer
retirement, a held host reader blocking detach, another actual module attachment
blocking replacement, fresh-device compute/readback, stale callback refusal,
insufficient candidate capabilities followed by confirmed discard and explicit
Retry, and healthy final teardown. Runtime GPU loss is deliberate, not physical
hardware removal. Mobile and real media-reader teardown need separate receipts.

## Browser final context and replacement

The Emscripten backend exposes `mgpuWebObserveContextRetirement`,
`mgpuWebContextRetirementStatus` and `mgpuWebReleaseContextObservation`. A token
captures the exact creation device/generation while ready and observes final
module teardown without retaining an attachment or GPU reference. Records are
bounded to 64, IDs are not reused, and failure remains failure. Dart retains the
exact original module/functions; an ambient module replacement cannot satisfy
the receipt. Teardown polling never occupies an Asyncify stack or a frame timer.
It does not use a lazy device getter after detach.

Returning one wrapper's attachment does not destroy another owner's device or
wait for that owner. Final completion joins the module receipt, destroys the
retained JS device and awaits its own loss promise. The root independently joins
host, GPU, external-reader and worker consumers before admitting a successor.

`DeviceRecoveryTarget` distinguishes native `SameAdapterRecoveryTarget` from
`BrowserRecoveryTarget`. The latter is an exact backend-owned, nonserializable
object with the original device's features and limits. Replacement requests a
fresh browser-selected device from the original module and revalidates those
requirements before publication. It promises no physical-adapter identity.
Lower maximum limits, missing features, or stricter alignment limits refuse.
The native LUID/backend contract is unchanged. Older backend assets without the
new observer exports can render but cannot claim browser recovery support.

Only confirmed final candidate retirement produces `DeviceRecoveryDiscarded`
and permits explicit root Retry. An unknown initialization/teardown error,
uncompleted reader, replaced module or failed receipt cannot be reset away.
Initialization and detach coalesce/fence their own creation generation; a last
detach joins its completed receipt before an ordinary close-then-init cycle.

## Final native context observation

`Minigpu.observeContextRetirement(deviceEpoch: ...)` registers a non-owning,
exact-generation observer while the wrapper still has its creation lease. Keep
that future across `destroy()`: local destruction returning only releases that
wrapper's attachment. The observer stays pending until other attachments and
creation pins leave and native context/external-cache teardown finishes. It does
not fence consumers or release them. Teardown exceptions retain quarantine and
complete the observer with failure, rather than marking the context reusable.

The C ABI `mgpuContextLeaseObserveRetirementAsync[ToPort]` admits at most 64
observers per device generation and retains no device, queue or attachment.
Callbacks execute outside the context lock. The Dart wrapper coalesces repeated
observation, preserves failed receipts and refuses a stale epoch. Unsupported
backends refuse; native completion is not browser-loss qualification.

The root must separately join worker/export/reader owners. A worker releases its
own local attachment on Stop, but must not await this final-context observer while
the root still holds its attachment and is waiting for Stop. Final context
completion attests native resource teardown, not an isolate's exit, an owned GPU
worker's destructor finishing, valid frame output or permission to reopen capture.

The focused native C++ retirement scenario checks bounded observation without
extra lease charges, device abortion with observers still pending, final-pin
release and explicit fresh initialization. The Dart native scenario checks the
same distinction across wrapper detach using the receipt instead of polling
reference counts. Native and Emscripten builds pass. These checks use deliberate
native device destruction, not physical driver removal.

## Host-owned same-adapter recovery

`recoverLostDevice(deviceEpoch: ...)` accepts only the current lost device. It
coalesces callers, joins registered owner/worker/reader receipts and final native
context retirement, then creates one replacement. On Windows the creation lease
captures the actual DXGI adapter LUID and D3D11/D3D12 backend; replacement must
match both. A name or vendor match is not proof. This identity is local to the
current boot and must not become a persistent/public device identifier. Owned
secondary contexts use a fresh handle instead of reviving a retired handle.

`onRecoveryChanged` reports draining, replacing, ready, failed or cancelled.
Embedder roots can subscribe to loss and schedule recovery after all synchronous
loss subscribers have fenced producers. Views may reprepare after ready only if
their own current authorization/source binding and replacement epoch still match.
The library does not restart media/capture, acquire permissions or replace an
application's authoritative state. An explicit `destroy()` during recovery waits
for cancellation and any candidate cleanup; it cannot race a late ready event.

Failed owner retirement stays charged and failed recovery is retained across
`init` and repeated recovery calls. Unsupported backends refuse. A wrong-identity
candidate is retired before reporting failure and is never published as ready.
Explicit Retry is available only after both original consumers and a conclusively
discarded candidate have retired; it is never a general quarantine reset.

Focused native cases cover graph pixels after deliberate loss with held readers,
coalesced recovery, same-adapter secondary replacement without resetting the
primary, and mismatched identity refusal. Controlled cases cover failed readers,
Close during replacement and healthy/stale requests. Native and Emscripten builds
pass. Windows D3D11 has actual-device evidence; D3D12 identity is compiled but not
hardware-qualified here. Browser replacement has the distinct capability-scoped
qualification above; mobile recovery and physical driver removal remain separate.

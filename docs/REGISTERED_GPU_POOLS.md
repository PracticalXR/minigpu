# Registered D3D11 output pools

`minigpu_external_pool.h` provides an additive local native integration API.
Creation requires an already-ready retained context with Dawn D3D11, Device5/Context4
and enabled texture-import/multithread protection. Capability observations and
creation pins must describe the same context/device generations. The API never
initializes, repairs or silently migrates a device. Other backends explicitly
refuse this first profile.

The context can be an explicitly owned handle or the trusted host's borrowed
display token from `mgpuDefaultContextRetainLease`. Keep that lease live through
the native consumer's synchronous retention. The pool then owns the exact
creation-device pin independently; no second device is initialized. See
[context ownership](REGISTERED_GPU_CONTEXTS.md).

Register separately typed NT texture and fence handles. Actual texture dimensions,
RGBA8/BGRA8 format, mip/layer/sample count, GPU-only storage, keyed-mutex/NT sharing
and adapter LUID must match the pool. Handles are borrowed during registration;
retained native objects survive duplicate closure and producer exit. Native IDs
and pointers are local objects, never wire identifiers or permission grants.
A trusted host selects fence purpose; native opening cannot establish authorized
encoder provenance. The integrating registrar enforces that provenance, exact
allocation identity, consent and route binding.

The source protocol is producer key 0 -> consumer key 1 -> key 0; the producer
releases key 1 before Submit. A separate native coordinator waits for the
producer fence with an event (SetEventOnCompletion) and polls only
AcquireSync(1, 0), checking S_OK explicitly; WAIT_TIMEOUT is nonnegative and is
not successful acquisition.
Foreign textures/mutexes never enter Dawn. The receiving worker copies into one
of three preallocated mutex-free host outputs, then signals and flushes a trusted
host fence. Only after actual copy completion and ReleaseSync(0) does the producer
receive SourceDone. Every SourceDone, including CANCELED or TIMEOUT before the
copy, leaves key 0 with the producer. There is one GPU copy and no CPU video
readback. Each host
allocation is imported once and has one reusable texture view.

SourceDone, OutputReady and OutputDone have separate lifetimes. AcquireOutput
claims an initial Dawn consumer and returns CONFLICT for an output that is
already acquired; RetainOutput permits four total consumers.
Dawn release fences the exact creation queue, including after logical retirement.
Native encoder consumers release against separately registered host-consumer
fences; producer-fence IDs are refused. Every consumer finishes before EndAccess
and output reuse. BindOutput validates creation ownership and queues in FIFO order.
A shader may not dispatch that binding after its consumer releases; replace the
binding or destroy the shader before pool release. Retaining a view alone does
not retain a consumer or make a recycled slot immutable.

Limits are 16 pools/1 GiB aggregate reservations, <=256 MiB per pool, three source
textures and three fences, three outputs, four consumers per output and 16 event
records. Admission reserves three terminal events and keeps one fault record
available. Full admission consumes no ownership. Sequence, lease token and each
producer fence's values strictly increase without accumulating replay history.
Host deadlines are at most five seconds. Pre-copy cancellation rolls back;
post-submit uncertainty faults admission and retains charged resources until
actual completion. Scheduling/task exceptions quarantine fixed storage instead
of terminating either worker. Destruction does not allocate synchronous jobs.
Drain events and consumers before final nonconcurrent Release, which joins the
coordinator and drains/cleans on a host worker, never the GPU worker. After device
loss or replacement, quarantined outputs no longer block Release.

The focused `minigpu_external_pool_test` passes on Windows x64 MSVC Release using
cached Dawn D3D11. A real separate producer supplies GPU-generated pixels sampled
on the receiving device. It covers wrong type/descriptor refusal, missing-signal
and held-mutex timeouts, constant allocation/import counts over eight frames,
output backpressure, independent encoder-fence completion and producer exit.
Deterministic Waiting/CopyQueued cancellation and cancellation/timeout of a real
submitted copy retain source, output and budget until a test-only trusted host
fence completes. All consumers also drain after context retirement. The same
target passes explicit D3D12 refusal. One independent review identified worker
exception handling and cancellation coverage; both were corrected before the
passing final gate.

This is not an encoder/publication, physical latency or device-fault receipt.
Authenticated registrar/tickets, full host adapters, real device/driver faults,
ARM64, other native backends and web require separate implementation/qualification.

September 15 duplicate-handle hardening retains one non-inheritable NT duplicate
per registered source/fence and imports through that exact retained handle.
`CompareObjectHandles` checks a fixed 96-entry table across live/retiring pools;
duplicate aliases refuse before import and cannot reset a fence or select another
purpose. Safe unregister releases the identity. No per-frame identity allocation,
comparison or import is added. The retained separate-process pixel/readiness/
completion gate passes in workspace `.tmp/x01-pool-kernel-identity-d3d11.log`.

This is kernel-object identity only. The separate `--fresh-identity-probe` mode
shows that newly created shared fence handles and an independently reopened
texture can refer to different kernel objects for the same GPU resource on the
current Windows/D3D11 driver. `.tmp/x01-pool-fresh-shared-handle-probe.log` records
that limit and actual cold-owner cleanup; it is an observation, not canonical
allocation, renderer or encoder qualification. Invasive alias probing was removed
from the retained pixel gate after it exposed stale samples there. Authenticated
external GPU admission still requires canonical allocation/alias and original
app/helper provenance verification. Producer-selected purposes cannot establish
host-consumer fence trust. The host GPU resource adapter remains unavailable.

## Host-issued producer pool

`mgpuExternalPoolCreateIssued` adds three shared render targets and a producer
fence to the existing three host outputs, inside the declared six-texture byte
reservation. `mgpuExternalPoolAcquireProducer` reserves one outstanding export
lease before exposing local borrowed NT handles and fixed source/fence IDs.
Trusted integration code must duplicate only into the original approved process.
Issued sources and the producer fence cannot be replaced or unregistered.

Retire preserves the export, and final Release refuses it independently of
source-copy/output-consumer completion. Only trusted completion with actual peer
release/exit proof consumes the exact export through
`mgpuExternalPoolReleaseProducerAfterPeerRelease`; timeout or failed delivery is
not proof. Host-consumer fence registration remains trusted host integration,
never a producer-selected privilege. No local handle or ID is a room permission.

The real separate-process `--issued` gate passes on Windows x64 MSVC Release in
workspace `.tmp/x01-pool-host-issued-d3d11.log`: direct producer rendering into
host-created allocations, exact sampled pixels, eight-frame reuse with six
native textures/three Dawn imports, injection refusal, backpressure, actual GPU
cancellation and export retention through retirement. Original process exit
releases the export once; actual host GPU completion and native cleanup restore
quota. The affected legacy gate passes, and explicit `MGPU_BACKEND=d3d12` use
refuses. Authenticated app/room grants, delivery and production renderer/encoder
adapters remain open; this native primitive does not install a room route.

The first issued run caught direct synchronous test readback overtaking queued
binding/dispatch. The gate now queues asynchronous readback after dispatch on
the creation worker. The earlier alias probe's stale pixels therefore do not
establish an alias-caused render fault; its separate kernel-identity limit still
holds. Test-only pixel readback adds no CPU readback to the pool.

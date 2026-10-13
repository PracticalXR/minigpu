# Retained GPU context ownership

The native context-handle API supports explicit creation-device leases for
asynchronous native consumers. Ordinary buffers and shaders still need to be
destroyed before their context handle is retired. A registered consumer holds a
lease through its actual GPU completion; submitting work or closing its owner
does not satisfy that lifetime.

`minigpu_context_lifetime.h` exposes retention, generation/current-state queries,
exact-queue completion waits and aggregate lifetime counters. Retention never
initializes or repairs a device. It requires a live handle; handle destruction
must not race that retention call. After retention, logical context retirement
immediately makes the lease non-current while preserving its creation device,
queue, instance and worker. The completion wait takes its own pin, so its caller
can release the borrowed lease afterward. Its native-port result follows the
existing `(token << 1) | success` format.

At most 64 charged context owners and 256 creation-state pins may remain live, including
retiring owners and pending work. Context creation returns null under pressure;
retention returns WOULD_BLOCK and clears its output. Physical destruction drains
and joins on one separate native retirement worker. A failed drain remains
quarantined and charged instead of freeing captured pointers. A pending or hung
GPU completion retains its pin; it is not converted into an invented success.
The application must quiesce all owners and leases before unloading the library.

Device-loss callbacks own atomic state for their original device generation.
They do not dereference a context/owner or acquire its lifecycle lock. Delayed
loss cannot invalidate a replacement device, and a loss observed during creation
cannot be overwritten by the later initialized flag. Completion waits retain the
original queue and instance, even if subsequent recovery changes the context.

`minigpu_external_pool.h` exposes the receiving-device facility query:
actual backend, adapter LUID, texture limit, D3D11 Device5/Context4 and enabled
Dawn texture-import/multithread protection. It checks the backend before using
Dawn's backend-specific helper. Other backends report explicit refusal. This is
device information, not permission to register or publish a room resource.
The process-global legacy context is not an owned context handle.

For trusted interop on the renderer's existing display device,
`mgpuDefaultContextRetainLease` returns a lease and borrowed pool-query token.
It retains one ready attachment, without initialization, adapter selection or
repair. Concurrent host scopes and native consumers share that attachment and
creation generation. The returned token is not an owned handle; owned-handle
initialization refuses it. Cold, final-detach and lock-bypass states refuse and
clear both outputs. The lease and a consumer's synchronously retained pin keep
the exact device/worker live even after the renderer detaches. The final pin
releases only this attachment on the retirement worker. The weak registry does
not keep an unused display device attached; queued/quarantined attachments share
the existing context-owner bound. Neither pointer is a wire identifier or grant.

`MinigpuFfi.withLiveExternalPoolContext` provides a synchronous int callback over
the installed query export and this borrowed token, releasing its temporary
lease on return or exception. The native consumer retains its own pin before
starting asynchronous work. `FfiSecondaryPlatform` exposes the corresponding
scope for an explicitly owned context.

The focused `minigpu_context_lifetime_test --default` scenario passes on Windows
x64 Dawn D3D11: cold refusal, actual display initialization, shared generations,
renderer/host-scope detach, retained native completion and recovered counters.
This does not qualify device replacement, web or other native backends.

The focused `minigpu_context_lifetime_test` target exercises the real native
device, bounded retention, retirement during initialization and exact-queue
completion draining after retirement. It passes on Windows x64 MSVC Release
with both D3D11 and D3D12 selected. Device replacement under fault, driver hangs,
module unloading and other native/web platforms require further qualification.
The [registered D3D11 pool](REGISTERED_GPU_POOLS.md) now implements native typed
registration, foreign readiness, three reusable host outputs and separate source,
Dawn and native-fence consumer completion. Its focused separate-process gate
passes. Authenticated registrar and room/encoder adapter integration remain open.

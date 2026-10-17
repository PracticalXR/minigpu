// Included after the production-owner CPU fakes: shares their real thread /
// registrar protocol without creating an engine, graphics context or image.
struct DispatchCallbackState {
  MGPUViewPrivateDispatch *request = nullptr;
  int calls = 0;
  uint64_t epoch = 0, id = 0;
};
static void dispatchCallback(uint64_t epoch, uint64_t id, void *userdata) {
  auto &state = *static_cast<DispatchCallbackState *>(userdata);
  ++state.calls; state.epoch = epoch; state.id = id;
  MGPUViewPrivateDispatchInfo info{};
  MGPUViewPrivateInfo preview{};
  MGPUViewPrivateRegistration *owner = reinterpret_cast<MGPUViewPrivateRegistration *>(uintptr_t(1));
  require(MGPUViewCollectPrivateDispatch(state.request, &info, sizeof(info),
      &owner, &preview, sizeof(preview)) == MGPU_VIEW_WOULD_BLOCK && !owner && !info.request_id,
      "callback entry is not callback return or output-ownership transfer");
  require(MGPUViewReleasePrivateDispatch(state.request) == MGPU_VIEW_WOULD_BLOCK,
          "callback cannot release its still-executing ticket");
}
static void pumpPrivateMessages() {
  MSG message{};
  while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
    TranslateMessage(&message); DispatchMessageW(&message);
  }
}
static MGPUViewPrivateRegistration *collectDispatch(DispatchCallbackState &state,
    MGPUViewPrivateInfo &preview, int expected) {
  MGPUViewPrivateDispatchInfo info{};
  MGPUViewPrivateRegistration *owner = nullptr;
  require(state.calls == 1 &&
      MGPUViewCollectPrivateDispatch(state.request, &info, sizeof(info),
        &owner, &preview, sizeof(preview)) == MGPU_VIEW_OK &&
      info.result == expected && info.source_epoch == state.epoch && info.request_id == state.id,
      "one checked transfer binds the actual operation result, source epoch and request");
  MGPUViewPrivateRegistration *again = reinterpret_cast<MGPUViewPrivateRegistration *>(uintptr_t(1));
  MGPUViewPrivateInfo empty{};
  require(MGPUViewCollectPrivateDispatch(state.request, &info, sizeof(info),
      &again, &empty, sizeof(empty)) == MGPU_VIEW_RETIRED && !again && !info.request_id,
      "a registration obligation cannot transfer twice");
  require(MGPUViewReleasePrivateDispatch(state.request) == MGPU_VIEW_OK,
          "request release follows execution and callback return");
  state.request = nullptr;
  return owner;
}
static void dispatchCases(MGPUViewTrustedHost *host, const MGPUViewHostBinding &qualification,
    Registrar &registrar, Raster &raster) {
  MGPUViewPrivateDispatcher *dispatcher = nullptr, *duplicate = nullptr;
  require(MGPUViewCreatePrivateDispatcher(host, &dispatcher) == 0 && dispatcher &&
      MGPUViewCreatePrivateDispatcher(host, &duplicate) == MGPU_VIEW_WOULD_BLOCK && !duplicate,
      "one actual headless platform-message dispatcher retains its original host");
  MGPUViewPrivateSourceBinding binding{40, 1, 1, GetTickCount64() + 60000,
      reinterpret_cast<void *>(&mgpuExternalPoolQueryCapabilities), 0};
  MGPUViewPrivateSource *source = nullptr;
  auto wrong = binding; wrong.original_minigpu_query = reinterpret_cast<void *>(uintptr_t(1));
  require(MGPUViewOpenPrivateSource(dispatcher, &wrong, &source) == MGPU_VIEW_UNSUPPORTED && !source,
      "wrong original provider refuses before any output/native pointer access");
  wrong = binding; wrong.valid_until_tick_ms = GetTickCount64();
  require(MGPUViewOpenPrivateSource(dispatcher, &wrong, &source) == MGPU_VIEW_RETIRED && !source,
      "already expired original deadline cannot open a source");
  require(MGPUViewOpenPrivateSource(dispatcher, &binding, &source) == 0 && source,
          "original source epoch and deadline open once");

  FakeOutput queued(registrar);
  std::array<DispatchCallbackState, 5> pending{};
  for (size_t i = 0; i < 4; ++i)
    require(MGPUViewPostPrivateRegister(source, &queued, queued.info.id,
        dispatchCallback, &pending[i], &pending[i].request) == 0,
        "bounded owned command posts without waiting for the platform");
  require(MGPUViewPostPrivateRegister(source, &queued, queued.info.id,
      dispatchCallback, &pending[4], &pending[4].request) == MGPU_VIEW_WOULD_BLOCK &&
      !pending[4].request && pending[4].calls == 0,
      "fifth pending request refuses without callback or ownership transfer");
  std::thread stopping([&] { require(MGPUViewFencePrivateSource(source) == 0,
                                    "worker Stop closes the source synchronously"); });
  stopping.join();
  require(MGPUViewReleasePrivateSource(source) == MGPU_VIEW_WOULD_BLOCK,
          "fenced source still counts every accepted queued callback");
  pumpPrivateMessages();
  MGPUViewPrivateInfo details{};
  for (size_t i = 0; i < 4; ++i)
    require(!collectDispatch(pending[i], details, MGPU_VIEW_RETIRED),
            "Stop before platform dequeue accepts no preview registration");
  require(!queued.acquires && MGPUViewReleasePrivateSource(source) == 0,
          "all pending commands drain before source release, without querying the output");
  source = nullptr;
  require(MGPUViewOpenPrivateSource(dispatcher, &binding, &source) == MGPU_VIEW_RETIRED && !source,
          "a closed source epoch cannot reopen under the same dispatcher");

  binding.source_epoch = 2;
  require(MGPUViewOpenPrivateSource(dispatcher, &binding, &source) == 0, "next original epoch");
  FakeOutput active(registrar);
  DispatchCallbackState ready;
  std::thread posting([&] {
    require(MGPUViewPostPrivateRegister(source, &active, active.info.id,
        dispatchCallback, &ready, &ready.request) == 0,
        "non-platform owner posts a real native registration command");
  });
  posting.join();
  pumpPrivateMessages();
  auto *reg = collectDispatch(ready, details, 0);
  require(reg && details.phase == MGPU_VIEW_ACTIVE, "worker adopts actual platform registration");
  auto *frame = registrar.frame(details.texture_id);
  require(frame && active.live == 1, "source-bound descriptor borrows the immutable output");
  require(MGPUViewFencePrivateSource(source) == 0 && !registrar.frame(details.texture_id) &&
      active.live == 1 && MGPUViewReleasePrivateSource(source) == MGPU_VIEW_WOULD_BLOCK,
      "source fence closes new descriptors while retaining an already admitted borrow");
  // Matching source checks precede any attempted retirement of the other owner.
  MGPUViewPrivateSource *other = nullptr; binding.source_epoch = 3;
  require(MGPUViewOpenPrivateSource(dispatcher, &binding, &other) == 0, "independent source");
  DispatchCallbackState mismatch;
  require(MGPUViewPostPrivateRetirement(other, reg, dispatchCallback, &mismatch,
      &mismatch.request) == MGPU_VIEW_INVALID && !mismatch.request && !mismatch.calls,
      "source identity prevents retiring another source's registration");
  require(MGPUViewFencePrivateSource(other) == 0 && MGPUViewReleasePrivateSource(other) == 0,
          "unused independent source drains");
  DispatchCallbackState retirement;
  require(MGPUViewPostPrivateRetirement(source, reg, dispatchCallback, &retirement,
      &retirement.request) == 0, "retirement remains available after source fencing");
  pumpPrivateMessages();
  require(!collectDispatch(retirement, details, 0), "Begin acknowledgement owns no new registration");
  MGPUViewPrivateReceipt receipt{};
  require(MGPUViewCollectPrivateRetirement(reg, &receipt, sizeof(receipt)) == MGPU_VIEW_WOULD_BLOCK &&
      !active.releases, "platform dispatch does not replace actual borrow/native-cleanup receipts");
  raster.run([&] { frame->release_callback(frame->release_context); });
  std::thread collecting([&] { collect(reg); }); collecting.join();
  require(active.releases == 1 && MGPUViewReleasePrivateSource(source) == 0,
          "source release follows checked removal, borrow drain and final registration release");

  binding.source_epoch = 4;
  require(MGPUViewOpenPrivateSource(dispatcher, &binding, &source) == 0, "racing source");
  FakeOutput raced(registrar);
  registrar.onRegister = [&] {
    std::thread revoke([&] { require(MGPUViewFencePrivateSource(source) == 0,
                                    "Stop can fence during platform registration"); });
    revoke.join();
  };
  DispatchCallbackState race;
  require(MGPUViewPostPrivateRegister(source, &raced, raced.info.id,
      dispatchCallback, &race, &race.request) == 0, "race request accepted");
  pumpPrivateMessages(); registrar.onRegister = {};
  reg = collectDispatch(race, details, MGPU_VIEW_RETIRED);
  require(reg && raced.acquired && !registrar.frame(details.texture_id),
          "error after accepted registration returns a closed cleanup owner, never drops it");
  retirement = {};
  require(MGPUViewPostPrivateRetirement(source, reg, dispatchCallback, &retirement,
      &retirement.request) == 0, "raced nonnull error owner still retires");
  pumpPrivateMessages(); collectDispatch(retirement, details, 0); collect(reg);
  require(MGPUViewReleasePrivateSource(source) == 0, "raced owner drains all native obligations");

  binding.source_epoch = 5;
  require(MGPUViewOpenPrivateSource(dispatcher, &binding, &source) == 0, "shutdown source");
  FakeOutput neverStarted(registrar); DispatchCallbackState shutdown;
  require(MGPUViewPostPrivateRegister(source, &neverStarted, neverStarted.info.id,
      dispatchCallback, &shutdown, &shutdown.request) == 0 &&
      MGPUViewClosePrivateDispatcher(dispatcher) == 0,
      "platform shutdown cancels accepted commands without executing them");
  require(!collectDispatch(shutdown, details, MGPU_VIEW_RETIRED) && !neverStarted.acquires,
          "cancelled command callback returns before ownership transfer");
  require(MGPUViewReleasePrivateSource(source) == 0 &&
      MGPUViewReleasePrivateDispatcher(dispatcher) == 0,
      "closed headless dispatcher releases only after every source and callback is drained");
  pumpPrivateMessages(); // Late wake messages contain no pointers and do nothing.

  require(MGPUViewCreatePrivateDispatcher(host, &dispatcher) == 0, "pre-reservation close dispatcher");
  binding.source_epoch = 6;
  require(MGPUViewOpenPrivateSource(dispatcher, &binding, &source) == 0, "pre-reservation source");
  FakeOutput beforeReserve(registrar); DispatchCallbackState before;
  beforeReserve.onQuery = [&] { require(MGPUViewClosePrivateDispatcher(dispatcher) == 0,
                                      "provider query reenters shutdown before reservation"); };
  require(MGPUViewPostPrivateRegister(source, &beforeReserve, beforeReserve.info.id,
      dispatchCallback, &before, &before.request) == 0, "pre-reservation command");
  pumpPrivateMessages();
  require(!collectDispatch(before, details, MGPU_VIEW_RETIRED) && !beforeReserve.acquires &&
      MGPUViewReleasePrivateSource(source) == 0 && MGPUViewReleasePrivateDispatcher(dispatcher) == 0,
      "monotonic shutdown refuses after query without a new registration or loan");

  for (int point = 0; point < 2; ++point) {
    // Quarantined fake resources intentionally stay heap-owned through process
    // exit, just like production's charged root. Never leave a dangling fake.
    MGPUViewTrustedHost *faultHost = host;
    if (point == 1)
      require(MGPUViewBindTrustedHost(&registrar, &qualification, &faultHost) == 0,
              "second qualified CPU host for independent shutdown fault");
    require(MGPUViewCreatePrivateDispatcher(faultHost, &dispatcher) == 0, "shutdown-fault dispatcher");
    binding.source_epoch = 7 + point;
    require(MGPUViewOpenPrivateSource(dispatcher, &binding, &source) == 0, "shutdown-fault source");
    auto *uncertain = new FakeOutput(registrar); DispatchCallbackState callback;
    auto shutdownNow = [&] { require(MGPUViewClosePrivateDispatcher(dispatcher) == 0,
                                     "native registration/loan call reenters shutdown"); };
    if (point == 0) registrar.onRegister = shutdownNow;
    else uncertain->onAcquire = shutdownNow;
    require(MGPUViewPostPrivateRegister(source, uncertain, uncertain->info.id,
        dispatchCallback, &callback, &callback.request) == 0, "shutdown-fault command");
    pumpPrivateMessages(); registrar.onRegister = {}; uncertain->onAcquire = {};
    reg = collectDispatch(callback, details, MGPU_VIEW_QUARANTINED);
    require(reg && details.phase == MGPU_VIEW_QUARANTINE &&
        MGPUViewBeginPrivateRetirement(reg) == MGPU_VIEW_QUARANTINED &&
        MGPUViewReleasePrivateRegistration(reg) == MGPU_VIEW_WOULD_BLOCK &&
        MGPUViewReleasePrivateSource(source) == MGPU_VIEW_WOULD_BLOCK &&
        MGPUViewReleasePrivateDispatcher(dispatcher) == MGPU_VIEW_WOULD_BLOCK,
        "shutdown quarantine is terminal and keeps registration/source/dispatcher charged");
    require(point == 0 ? !uncertain->acquires : uncertain->acquired && uncertain->quarantined,
        "shutdown before acquire skips it; a loan returned across shutdown is quarantined");
  }
}

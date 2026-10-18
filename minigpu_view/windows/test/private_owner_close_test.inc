// Actual native close owner/message dispatch; only the existing registrar and
// producer are CPU fakes. No Flutter engine, GPU or capture is created.
static void ownerCloseCases(const MGPUViewHostBinding &binding, Registrar &registrar) {
  int core = 0, plugin = 0, otherCore = 0, otherPlugin = 0;
  std::unique_ptr<minigpu_view::RegistrarAssociationOwner> owner, otherOwner;
  require(minigpu_view::RegistrarAssociationOwner::Reserve(&core, &plugin, owner) == 0 &&
      minigpu_view::PublishRegistrarAssociation(owner->publication(), &registrar) == 0,
      "close owner uses privately published actual pair");
  MGPUViewRegistrarAssociation *association = nullptr, *otherAssociation = nullptr;
  require(MGPUViewAcquireRegistrarAssociation(&core, &association) == 0 &&
      minigpu_view::RegistrarAssociationOwner::Reserve(&otherCore, &otherPlugin, otherOwner) == 0 &&
      minigpu_view::PublishRegistrarAssociation(otherOwner->publication(), &registrar) == 0 &&
      MGPUViewAcquireRegistrarAssociation(&otherCore, &otherAssociation) == 0,
      "independent association for exact ownership refusal");
  MGPUViewPrivateOwnerCloseInfo info{};
  MGPUViewPrivateOwnerCloseReceipt receipt{};
  std::thread wrongThread([&] {
    MGPUViewPrivateOwnerClose *entry = reinterpret_cast<MGPUViewPrivateOwnerClose *>(uintptr_t(1));
    require(MGPUViewCreatePrivateOwnerClose(association, &entry, &info, sizeof(info)) ==
        MGPU_VIEW_WOULD_BLOCK && !entry && !info.request_id,
        "worker cannot manufacture the platform cleanup endpoint");
  });
  wrongThread.join();
  for (int kind = 0; kind < 3; ++kind) {
    MGPUViewPrivateOwnerClose *closeOwner = nullptr, *wrongClose = nullptr;
    MGPUViewTrustedHost *host = nullptr;
    MGPUViewPrivateDispatcher *dispatcher = nullptr;
    require(MGPUViewCreatePrivateOwnerClose(association, &closeOwner, &info, sizeof(info)) == 0 &&
        info.struct_size == 48 && info.version == 1 && info.request_id &&
        info.association_generation && info.wake_event && !info.has_host && !info.has_dispatcher,
        "platform cleanup endpoint exists BEFORE host bind");
    const auto requestId = info.request_id, generation = info.association_generation;
    if (kind) {
      require(MGPUViewBindAssociatedTrustedHost(association, &binding, &host) == 0,
              "cleanup metadata did not consume the unique host binding");
      require(MGPUViewCreatePrivateOwnerClose(otherAssociation, &wrongClose, &info, sizeof(info)) == 0 &&
          MGPUViewBeginPrivateOwnerClose(wrongClose, host, nullptr) == MGPU_VIEW_INVALID,
          "foreign association cannot close this host");
      require(MGPUViewBeginPrivateOwnerClose(wrongClose, nullptr, nullptr) == 0,
              "refused close accepted no host and can close its own unused endpoint");
      pumpPrivateMessages();
      require(MGPUViewCollectPrivateOwnerClose(wrongClose, &receipt, sizeof(receipt)) == 0 &&
          !receipt.had_host && MGPUViewReleasePrivateOwnerClose(wrongClose) == 0,
          "wrong-association request drains without touching its rejected host");
      if (kind == 2) {
        require(MGPUViewCreatePrivateDispatcher(host, &dispatcher) == 0,
                "complete dormant parent includes owned dispatcher");
        MGPUViewPrivateSourceBinding sourceBinding{40,1,1,GetTickCount64()+60000,
            reinterpret_cast<void *>(&mgpuExternalPoolQueryCapabilities),0};
        MGPUViewPrivateSource *source = nullptr;
        require(MGPUViewOpenPrivateSource(dispatcher, &sourceBinding, &source) == 0 &&
            MGPUViewBeginPrivateOwnerClose(closeOwner, host, dispatcher) == MGPU_VIEW_WOULD_BLOCK,
            "a parent cannot close while another source child is alive");
        require(MGPUViewFencePrivateSource(source) == 0 && MGPUViewReleasePrivateSource(source) == 0,
                "child source drains before final-parent closure");
      }
    }
    // A prospective dispatcher creator has passed its early open check. Pause
    // at the same production admission gate used just before publication.
    const auto prospectiveCreator = kind == 1 ? RetainPrivateHost(host) : nullptr;
    if (prospectiveCreator) require(PrivateHostOpen(prospectiveCreator), "creator's early open check");
    std::thread closing([&] {
      require(MGPUViewBeginPrivateOwnerClose(closeOwner, host, dispatcher) == 0,
              "worker queues fixed native parent cleanup");
      require(MGPUViewCollectPrivateOwnerClose(closeOwner, &receipt, sizeof(receipt)) ==
          MGPU_VIEW_WOULD_BLOCK && !receipt.request_id &&
          MGPUViewReleasePrivateOwnerClose(closeOwner) == MGPU_VIEW_WOULD_BLOCK,
          "without platform pumping no close receipt or final release exists");
      if (host) require(MGPUViewReleaseTrustedHost(host) == MGPU_VIEW_WOULD_BLOCK,
                        "queued platform close is not completed host close");
    });
    closing.join();
    if (prospectiveCreator) {
      require(!TryPrivateHostAddChild(prospectiveCreator),
              "late dispatcher admission cannot cross the worker's reserved host-close fence");
      MGPUViewPrivateDispatcher *late = nullptr;
      require(MGPUViewCreatePrivateDispatcher(host, &late) == MGPU_VIEW_RETIRED && !late,
              "closed host publishes no new dispatcher");
    }
    pumpPrivateMessages();
    require(MGPUViewCollectPrivateOwnerClose(closeOwner, &receipt, sizeof(receipt)) == 0 &&
        receipt.struct_size == 40 && receipt.version == 1 && !receipt.result && !receipt.reserved &&
        receipt.request_id == requestId && receipt.association_generation == generation &&
        receipt.had_host == static_cast<uint32_t>(kind != 0) &&
        receipt.had_dispatcher == static_cast<uint32_t>(kind == 2),
        "receipt follows actual platform close return and both window removals");
    if (dispatcher) require(MGPUViewReleasePrivateDispatcher(dispatcher) == 0, "dispatcher final release");
    if (host) require(MGPUViewReleaseTrustedHost(host) == 0, "host final release");
    require(MGPUViewReleasePrivateOwnerClose(closeOwner) == 0, "close owner final release");
  }
  std::array<MGPUViewPrivateOwnerClose *,32> capacity{};
  for (auto &entry : capacity)
    require(MGPUViewCreatePrivateOwnerClose(association, &entry, &info, sizeof(info)) == 0,
            "bounded close owner capacity");
  MGPUViewPrivateOwnerClose *overflow = reinterpret_cast<MGPUViewPrivateOwnerClose *>(uintptr_t(1));
  require(MGPUViewCreatePrivateOwnerClose(association, &overflow, &info, sizeof(info)) ==
      MGPU_VIEW_WOULD_BLOCK && !overflow && !info.request_id, "33rd cleanup owner accepts no obligation");
  for (auto *entry : capacity)
    require(MGPUViewBeginPrivateOwnerClose(entry, nullptr, nullptr) == 0, "close empty endpoint");
  pumpPrivateMessages();
  for (auto *entry : capacity)
    require(MGPUViewCollectPrivateOwnerClose(entry, &receipt, sizeof(receipt)) == 0 &&
        MGPUViewReleasePrivateOwnerClose(entry) == 0, "bounded close owners drain");
  MGPUViewPrivateOwnerClose *lostEndpoint = nullptr;
  require(MGPUViewCreatePrivateOwnerClose(association, &lostEndpoint, &info, sizeof(info)) == 0,
          "owned endpoint for native loss refusal");
  const auto window = FindWindowExW(HWND_MESSAGE, nullptr, L"MinigpuView.PrivateOwnerClose.v1", L"");
  require(window && DestroyWindow(window), "fault-injected native endpoint removal");
  require(MGPUViewBeginPrivateOwnerClose(lostEndpoint, nullptr, nullptr) == MGPU_VIEW_QUARANTINED &&
      MGPUViewCollectPrivateOwnerClose(lostEndpoint, &receipt, sizeof(receipt)) == MGPU_VIEW_QUARANTINED &&
      !receipt.request_id && MGPUViewReleasePrivateOwnerClose(lostEndpoint) == MGPU_VIEW_WOULD_BLOCK,
      "lost platform endpoint retains its charged owner without a receipt");
  owner.reset(); otherOwner.reset();
  require(MGPUViewReleaseRegistrarAssociation(association) == 0 &&
      MGPUViewReleaseRegistrarAssociation(otherAssociation) == 0, "close-only metadata acquisition releases");
}

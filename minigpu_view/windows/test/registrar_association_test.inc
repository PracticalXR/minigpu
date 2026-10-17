// The real association/bind/dispatch implementations, with the existing CPU
// registrar/provider protocol. These identities are local test tokens, not an
// actual engine, engine-retention receipt or deployed qualification profile.
struct AssociationVerification {
  void *core = nullptr;
  std::function<void()> during;
};
static int verifyAssociation(void *identity, const MGPUViewHostBinding *binding, void *) {
  auto *context = static_cast<AssociationVerification *>(binding->verifier_userdata);
  require(context && context->core == identity,
          "associated verifier receives the exact core C registrar, never the C++ wrapper");
  if (context && context->during) context->during();
  return qualified ? MGPU_VIEW_OK : MGPU_VIEW_UNSUPPORTED;
}
static void associationCases(MGPUViewHostBinding binding, Registrar &registrar, Raster &raster) {
  using minigpu_view::RegistrarAssociationOwner;
  using minigpu_view::PublishRegistrarAssociation;
  int core = 0, plugin = 0, otherPlugin = 0;
  std::unique_ptr<RegistrarAssociationOwner> owner, duplicate;
  MGPUViewRegistrarAssociation *handle = nullptr;
  require(RegistrarAssociationOwner::Reserve(&core, &plugin, owner) == MGPU_VIEW_OK && owner,
          "actual pair first reserves unpublished metadata");
  require(MGPUViewAcquireRegistrarAssociation(&core, &handle) == MGPU_VIEW_NOT_READY && !handle &&
      RegistrarAssociationOwner::Reserve(&core, &plugin, duplicate) == MGPU_VIEW_WOULD_BLOCK && !duplicate,
      "unpublished registration cannot acquire or reenter plugin installation");
  const auto publication = owner->publication();
  require(PublishRegistrarAssociation(publication, &registrar) == MGPU_VIEW_OK &&
      PublishRegistrarAssociation(publication, &registrar) == MGPU_VIEW_OK &&
      MGPUViewAcquireRegistrarAssociation(&core, &handle) == MGPU_VIEW_OK && handle,
      "private publication makes the exact pair discoverable");
  MGPUViewRegistrarAssociationInfo info{};
  require(MGPUViewGetRegistrarAssociationInfo(handle, &info, sizeof(info)) == MGPU_VIEW_OK &&
      info.struct_size == 32 && info.version == 1 && info.phase == MGPU_VIEW_ASSOCIATION_OPEN &&
      info.platform_thread_id == GetCurrentThreadId() && info.association_generation && !info.reserved,
      "association info exposes generation/thread metadata without a wrapper pointer");
  const auto firstGeneration = info.association_generation;
  require(RegistrarAssociationOwner::Reserve(&core, &plugin, duplicate) == MGPU_VIEW_OK && !duplicate &&
      RegistrarAssociationOwner::Reserve(&core, &otherPlugin, duplicate) == MGPU_VIEW_INVALID && !duplicate,
      "exact published duplicate is a no-op, conflicting native registration refuses");
  Registrar otherRegistrar(raster);
  require(PublishRegistrarAssociation(publication, &otherRegistrar) == MGPU_VIEW_INVALID,
          "published association cannot change its original wrapper");
  std::thread wrongThread([&] {
    MGPUViewRegistrarAssociation *wrong = reinterpret_cast<MGPUViewRegistrarAssociation *>(uintptr_t(1));
    MGPUViewTrustedHost *host = reinterpret_cast<MGPUViewTrustedHost *>(uintptr_t(1));
    require(MGPUViewAcquireRegistrarAssociation(&core, &wrong) == MGPU_VIEW_WOULD_BLOCK && !wrong &&
        MGPUViewBindAssociatedTrustedHost(handle, &binding, &host) == MGPU_VIEW_WOULD_BLOCK && !host &&
        RegistrarAssociationOwner::Reserve(&core, &plugin, duplicate) == MGPU_VIEW_WOULD_BLOCK,
        "wrong thread cannot acquire, bind, or reinstall the pair");
    MGPUViewRegistrarAssociationInfo copied{};
    require(MGPUViewGetRegistrarAssociationInfo(handle, &copied, sizeof(copied)) == MGPU_VIEW_OK &&
        copied.association_generation == firstGeneration, "serialized worker may copy metadata");
  });
  wrongThread.join();
  std::array<MGPUViewRegistrarAssociation *, 128> handles{};
  handles[0] = handle;
  for (size_t i = 1; i < handles.size(); ++i)
    require(MGPUViewAcquireRegistrarAssociation(&core, &handles[i]) == MGPU_VIEW_OK,
            "bounded acquisition capacity");
  MGPUViewRegistrarAssociation *overflow = nullptr;
  require(MGPUViewAcquireRegistrarAssociation(&core, &overflow) == MGPU_VIEW_WOULD_BLOCK && !overflow,
          "129th acquisition accepts no ownership");
  for (size_t i = 1; i < handles.size(); ++i)
    require(MGPUViewReleaseRegistrarAssociation(handles[i]) == MGPU_VIEW_OK, "acquisition release");

  binding.verify = verifyAssociation;
  AssociationVerification verification;
  verification.core = &core;
  binding.verifier_userdata = &verification;
  MGPUViewTrustedHost *host = nullptr, *second = nullptr;
  qualified = false;
  require(MGPUViewBindAssociatedTrustedHost(handle, &binding, &host) == MGPU_VIEW_UNSUPPORTED && !host,
          "association does not supply independent renderer qualification");
  qualified = true;
  verification.during = [&] { throw 1; };
  require(MGPUViewBindAssociatedTrustedHost(handle, &binding, &host) == MGPU_VIEW_INTERNAL && !host,
          "verifier exception releases a never-published binding reservation");
  verification.during = [&] {
    require(MGPUViewBindAssociatedTrustedHost(handle, &binding, &second) == MGPU_VIEW_WOULD_BLOCK && !second,
            "binding reserves before foreign verifier reentry");
  };
  require(MGPUViewBindAssociatedTrustedHost(handle, &binding, &host) == MGPU_VIEW_OK && host &&
      MGPUViewBindAssociatedTrustedHost(handle, &binding, &second) == MGPU_VIEW_WOULD_BLOCK && !second,
      "one associated host at a time, after independent CPU fake qualification");
  verification.during = nullptr;
  FakeOutput output(registrar);
  MGPUViewPrivateInfo details{};
  auto *registration = add(host, &output, details);
  const auto *frame = registrar.frame(details.texture_id);
  require(frame, "associated host uses its original wrapper for the actual owner protocol");
  raster.run([&] { frame->release_callback(frame->release_context); });
  require(MGPUViewBeginPrivateRetirement(registration) == MGPU_VIEW_OK, "associated removal begins");
  collect(registration);
  MGPUViewPrivateDispatcher *dispatcher = nullptr;
  require(MGPUViewCreatePrivateDispatcher(host, &dispatcher) == MGPU_VIEW_OK && dispatcher,
          "dispatcher retains the associated host");
  require(MGPUViewCloseTrustedHost(host) == MGPU_VIEW_OK &&
      MGPUViewReleaseTrustedHost(host) == MGPU_VIEW_WOULD_BLOCK &&
      MGPUViewBindAssociatedTrustedHost(handle, &binding, &second) == MGPU_VIEW_WOULD_BLOCK,
      "closed host still owns the binding until its dispatcher drains");
  require(MGPUViewClosePrivateDispatcher(dispatcher) == MGPU_VIEW_OK &&
      MGPUViewReleasePrivateDispatcher(dispatcher) == MGPU_VIEW_OK &&
      MGPUViewReleaseTrustedHost(host) == MGPU_VIEW_OK, "children drain before associated host release");
  host = nullptr;

  // Close while a previously accepted registration command is still queued.
  require(MGPUViewBindAssociatedTrustedHost(handle, &binding, &host) == MGPU_VIEW_OK &&
      MGPUViewCreatePrivateDispatcher(host, &dispatcher) == MGPU_VIEW_OK, "binding slot becomes reusable");
  MGPUViewPrivateSourceBinding sourceBinding{40, 1, 1, GetTickCount64() + 60000,
      reinterpret_cast<void *>(&mgpuExternalPoolQueryCapabilities), 0};
  MGPUViewPrivateSource *source = nullptr;
  require(MGPUViewOpenPrivateSource(dispatcher, &sourceBinding, &source) == MGPU_VIEW_OK,
          "associated source opens");
  FakeOutput queued(registrar);
  DispatchCallbackState pending;
  require(MGPUViewPostPrivateRegister(source, &queued, queued.info.id, dispatchCallback, &pending,
      &pending.request) == MGPU_VIEW_OK, "associated queued command is owned");
  owner.reset(); // Plugin's first shutdown fence; wrapper is still retained by caller's engine.
  require(PublishRegistrarAssociation(publication, &registrar) == MGPU_VIEW_RETIRED,
          "reentrant or late publication cannot reopen plugin closure");
  const auto registrationCount = registrar.registrations;
  pumpPrivateMessages();
  require(!collectDispatch(pending, details, MGPU_VIEW_RETIRED) && !queued.acquires &&
      registrar.registrations == registrationCount, "queued work observes association closure before registrar use");
  require(MGPUViewFencePrivateSource(source) == MGPU_VIEW_OK &&
      MGPUViewReleasePrivateSource(source) == MGPU_VIEW_OK &&
      MGPUViewClosePrivateDispatcher(dispatcher) == MGPU_VIEW_OK &&
      MGPUViewReleasePrivateDispatcher(dispatcher) == MGPU_VIEW_OK &&
      MGPUViewCloseTrustedHost(host) == MGPU_VIEW_OK &&
      MGPUViewReleaseTrustedHost(host) == MGPU_VIEW_OK, "closed association retains drainable CPU ownership");
  host = nullptr;
  require(MGPUViewGetRegistrarAssociationInfo(handle, &info, sizeof(info)) == MGPU_VIEW_OK &&
      info.phase == MGPU_VIEW_ASSOCIATION_CLOSED, "old acquired token remains permanently closed");

  require(RegistrarAssociationOwner::Reserve(&core, &plugin, owner) == MGPU_VIEW_OK && owner &&
      PublishRegistrarAssociation(owner->publication(), &registrar) == MGPU_VIEW_OK,
      "same core address may reincarnate under a new generation");
  MGPUViewRegistrarAssociation *fresh = nullptr;
  require(MGPUViewAcquireRegistrarAssociation(&core, &fresh) == MGPU_VIEW_OK &&
      MGPUViewGetRegistrarAssociationInfo(fresh, &info, sizeof(info)) == MGPU_VIEW_OK &&
      info.association_generation > firstGeneration &&
      MGPUViewBindAssociatedTrustedHost(handle, &binding, &host) == MGPU_VIEW_RETIRED && !host,
      "new incarnation does not revive an old handle");
  const auto freshPublication = owner->publication();
  verification.during = [&] {
    owner.reset();
    require(PublishRegistrarAssociation(freshPublication, &registrar) == MGPU_VIEW_RETIRED,
            "foreign qualification callback cannot revive its deleted association owner");
  };
  require(MGPUViewBindAssociatedTrustedHost(fresh, &binding, &host) == MGPU_VIEW_RETIRED && !host,
          "verifier-triggered closure refuses late host publication");
  verification.during = nullptr;
  require(MGPUViewReleaseRegistrarAssociation(handle) == MGPU_VIEW_OK &&
      MGPUViewReleaseRegistrarAssociation(fresh) == MGPU_VIEW_OK, "closed acquired tokens release after bind unwinds");

  // Closed states remain bounded/charged while external handles retain them.
  std::array<int, 17> cores{}, plugins{};
  std::array<MGPUViewRegistrarAssociation *, 16> retained{};
  for (size_t i = 0; i < retained.size(); ++i) {
    require(RegistrarAssociationOwner::Reserve(&cores[i], &plugins[i], owner) == MGPU_VIEW_OK && owner &&
        PublishRegistrarAssociation(owner->publication(), &registrar) == MGPU_VIEW_OK &&
        MGPUViewAcquireRegistrarAssociation(&cores[i], &retained[i]) == MGPU_VIEW_OK,
        "bounded retained closed state");
    owner.reset();
  }
  require(RegistrarAssociationOwner::Reserve(&cores[16], &plugins[16], owner) == MGPU_VIEW_NOT_READY && !owner,
          "17th closed retained association cannot evict an old owner");
  require(MGPUViewReleaseRegistrarAssociation(retained[0]) == MGPU_VIEW_OK &&
      RegistrarAssociationOwner::Reserve(&cores[16], &plugins[16], owner) == MGPU_VIEW_OK && owner,
      "final retained-handle release frees exactly one association slot");
  owner.reset();
  for (size_t i = 1; i < retained.size(); ++i)
    require(MGPUViewReleaseRegistrarAssociation(retained[i]) == MGPU_VIEW_OK, "closed capacity drains");
}

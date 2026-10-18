#ifndef MINIGPU_VIEW_REGISTRAR_ASSOCIATION_OWNER_H_
#define MINIGPU_VIEW_REGISTRAR_ASSOCIATION_OWNER_H_
#include "include/minigpu_view/minigpu_view_registrar_association.h"
#include <flutter/texture_registrar.h>
#include <memory>
namespace minigpu_view {
struct RegistrarAssociationState;
class RegistrarAssociationOwner final {
public:
  ~RegistrarAssociationOwner();
  // OK+null means the exact registration is already published. NOT_READY is
  // optional-profile capacity failure; ordinary plugin registration may proceed
  // without private association support. Conflict/reentry must not reregister.
  static int Reserve(void *core, void *original_plugin_registrar,
                     std::unique_ptr<RegistrarAssociationOwner> &);
  void Close() noexcept;
  std::shared_ptr<RegistrarAssociationState> publication() const { return state_; }
  RegistrarAssociationOwner(const RegistrarAssociationOwner &) = delete;
  RegistrarAssociationOwner &operator=(const RegistrarAssociationOwner &) = delete;
private:
  explicit RegistrarAssociationOwner(std::shared_ptr<RegistrarAssociationState>);
  std::shared_ptr<RegistrarAssociationState> state_;
};
// Private plugin registration scope only. A snapshot survives reentrant owner
// teardown; publication can never reopen its closed state.
int PublishRegistrarAssociation(const std::shared_ptr<RegistrarAssociationState> &,
                                flutter::TextureRegistrar *);
class RegistrarAssociationBinding final {
public:
  ~RegistrarAssociationBinding();
  bool open() const noexcept;
  uint32_t platformThreadId() const noexcept;
  void *coreRegistrarOnPlatform() const noexcept;
  // Private implementation access, never exported/copy-out ABI. Caller retains
  // its real engine and invokes wrapper operations only on the platform thread.
  flutter::TextureRegistrar *registrarOnPlatform() const noexcept;
private:
  explicit RegistrarAssociationBinding(std::shared_ptr<RegistrarAssociationState>);
  std::shared_ptr<RegistrarAssociationState> state_;
  bool reserved_ = false;
  friend int BeginRegistrarAssociationBinding(MGPUViewRegistrarAssociation *,
      std::shared_ptr<RegistrarAssociationBinding> &);
  friend bool SameRegistrarAssociation(const std::shared_ptr<RegistrarAssociationBinding> &,
                                      MGPUViewRegistrarAssociation *);
};
int BeginRegistrarAssociationBinding(MGPUViewRegistrarAssociation *,
    std::shared_ptr<RegistrarAssociationBinding> &);
// Private metadata retention for a cleanup endpoint; no unique host-binding
// slot is consumed. The extra acquisition remains counted through release.
int RetainRegistrarAssociationForClose(MGPUViewRegistrarAssociation *, MGPUViewRegistrarAssociation **);
bool SameRegistrarAssociation(const std::shared_ptr<RegistrarAssociationBinding> &,
                             MGPUViewRegistrarAssociation *);
}
#endif

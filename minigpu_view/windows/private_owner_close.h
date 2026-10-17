#ifndef MINIGPU_VIEW_PRIVATE_OWNER_CLOSE_IMPL_H_
#define MINIGPU_VIEW_PRIVATE_OWNER_CLOSE_IMPL_H_
#include "private_preview_dispatcher.h"
#include "include/minigpu_view/minigpu_view_private_dispatch.h"
#include "registrar_association.h"
namespace minigpu_view::private_preview {
// Atomic no-foreign-call reservation against registration creation. h==null
// is valid only for the cleanup endpoint's empty preparation case.
int ReservePrivateHostClose(const std::shared_ptr<Host> &, MGPUViewRegistrarAssociation *,
                           uint64_t request, uint32_t expected_children);
int CloseRetainedPrivateHost(const std::shared_ptr<Host> &);
}
namespace minigpu_view::private_dispatch {
struct Dispatcher;
std::shared_ptr<Dispatcher> RetainPrivateDispatcher(MGPUViewPrivateDispatcher *);
int ReservePrivateOwnerPair(const std::shared_ptr<private_preview::Host> &,
    const std::shared_ptr<Dispatcher> &, MGPUViewRegistrarAssociation *, uint64_t request);
int CloseRetainedPrivateDispatcher(const std::shared_ptr<Dispatcher> &);
}
#endif

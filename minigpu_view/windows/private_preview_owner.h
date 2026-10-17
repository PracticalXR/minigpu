#ifndef MINIGPU_VIEW_PRIVATE_PREVIEW_OWNER_H_
#define MINIGPU_VIEW_PRIVATE_PREVIEW_OWNER_H_
#include <flutter/texture_registrar.h>
#include "include/minigpu_view/minigpu_view_private_preview.h"
namespace minigpu_view {
// Plugin shutdown must close strict owners before its registrar becomes invalid.
void ClosePrivatePreviewRegistrar(flutter::TextureRegistrar *) noexcept;
}
#endif

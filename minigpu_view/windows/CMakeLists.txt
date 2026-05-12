cmake_minimum_required(VERSION 3.14)

# Project-level configuration.
set(PROJECT_NAME "minigpu_view")
project(${PROJECT_NAME} LANGUAGES CXX)

# Explicitly opt in to modern CMake behaviors to avoid warnings with recent
# versions of CMake.
cmake_policy(VERSION 3.14...3.25)

# This value is used when generating builds using this plugin, so it must
# not be changed.
set(PLUGIN_NAME "minigpu_view_plugin")

list(APPEND PLUGIN_SOURCES
  "minigpu_view_plugin.cpp"
  "minigpu_view_plugin_impl.h"
  "d3d11_texture_handler.cpp"
  "d3d11_texture_handler.h"
)

add_library(${PLUGIN_NAME} SHARED
  "include/minigpu_view/minigpu_view_plugin.h"
  ${PLUGIN_SOURCES}
)

apply_standard_settings(${PLUGIN_NAME})

set_target_properties(${PLUGIN_NAME} PROPERTIES
  CXX_VISIBILITY_PRESET hidden)
target_compile_definitions(${PLUGIN_NAME} PRIVATE FLUTTER_PLUGIN_IMPL)

target_include_directories(${PLUGIN_NAME} INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(${PLUGIN_NAME} PRIVATE
  flutter
  flutter_wrapper_plugin
  d3d11
  dxgi
)

# List of absolute paths to libraries that should be bundled with the plugin.
set(minigpu_view_bundled_libraries
  ""
  PARENT_SCOPE
)

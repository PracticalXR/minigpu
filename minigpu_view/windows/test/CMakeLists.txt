cmake_minimum_required(VERSION 3.14)
project(minigpu_view_private_boundary LANGUAGES C CXX)
if(NOT WIN32 OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "The private preview V1 boundary requires Windows x64")
endif()
if(NOT DEFINED FLUTTER_WRAPPER_INCLUDE OR NOT DEFINED FLUTTER_EMBEDDER_INCLUDE OR
   NOT DEFINED MINIGPU_NATIVE_INCLUDE)
  message(FATAL_ERROR "Pass installed Flutter headers and the native producer ABI include directory")
endif()
add_executable(minigpu_view_private_boundary
  "../private_preview_owner.cpp" "../private_preview_dispatcher.cpp" "../registrar_association.cpp"
  "../private_owner_close.cpp" "private_preview_owner_test.cpp"
  "private_preview_abi_c.c")
target_compile_features(minigpu_view_private_boundary PRIVATE cxx_std_17)
target_compile_definitions(minigpu_view_private_boundary PRIVATE MGPU_VIEW_PRIVATE_PREVIEW_IMPL)
target_include_directories(minigpu_view_private_boundary PRIVATE
  "${FLUTTER_WRAPPER_INCLUDE}" "${FLUTTER_EMBEDDER_INCLUDE}" "${MINIGPU_NATIVE_INCLUDE}")

# Actual production owner DLL for cold SDK FFI interop. No engine/provider fake,
# image, device or registration is constructed merely by loading/querying it.
# Test artifact only; it is not the installed Flutter plugin or qualification.
add_library(minigpu_view_private_cpu_fixture SHARED
  "../private_preview_owner.cpp" "../private_preview_dispatcher.cpp" "../registrar_association.cpp"
  "../private_owner_close.cpp")
target_compile_features(minigpu_view_private_cpu_fixture PRIVATE cxx_std_17)
target_include_directories(minigpu_view_private_cpu_fixture PRIVATE
  "${FLUTTER_WRAPPER_INCLUDE}" "${FLUTTER_EMBEDDER_INCLUDE}")

# Compile AND link the real C plugin entry and its DLL-local wrapper manager.
# This target is never loaded or run: it proves registration wiring against the
# installed headers/import library without launching an engine, D3D or capture.
get_filename_component(FLUTTER_WRAPPER_DIR "${FLUTTER_WRAPPER_INCLUDE}/.." ABSOLUTE)
add_library(minigpu_view_registration_link SHARED
  "../minigpu_view_plugin.cpp" "../d3d11_texture_handler.cpp"
  "../private_preview_owner.cpp" "../private_preview_dispatcher.cpp" "../registrar_association.cpp"
  "../private_owner_close.cpp"
  "${FLUTTER_WRAPPER_DIR}/core_implementations.cc"
  "${FLUTTER_WRAPPER_DIR}/standard_codec.cc"
  "${FLUTTER_WRAPPER_DIR}/plugin_registrar.cc")
target_compile_features(minigpu_view_registration_link PRIVATE cxx_std_17)
target_compile_options(minigpu_view_registration_link PRIVATE /W4 /WX /wd4100 /EHsc)
target_compile_definitions(minigpu_view_registration_link PRIVATE FLUTTER_PLUGIN_IMPL _HAS_EXCEPTIONS=0)
target_include_directories(minigpu_view_registration_link PRIVATE
  "${FLUTTER_WRAPPER_INCLUDE}" "${FLUTTER_EMBEDDER_INCLUDE}")
target_link_libraries(minigpu_view_registration_link PRIVATE
  "${FLUTTER_EMBEDDER_INCLUDE}/flutter_windows.dll.lib" d3d11 dxgi)

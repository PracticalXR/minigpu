set(EXT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/external" CACHE STRING "Path to the local source directory")
set(CMAKE_CXX_STANDARD 20)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/download.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/print_target.cmake")

download_repository("gpu" "https://github.com/MichealReed/gpu.cpp" "${EXT_SOURCE_DIR}/gpu" "minigpu")

set(GPU_INCLUDE_DIR "${EXT_SOURCE_DIR}/gpu")

# Add sources
set(GPU_SOURCES
    "${GPU_INCLUDE_DIR}/gpu.cpp"
    "${GPU_INCLUDE_DIR}/numeric_types/half.cpp"
)

# Add headers
set(GPU_HEADERS
    "${GPU_INCLUDE_DIR}/gpu.hpp"
    "${GPU_INCLUDE_DIR}/utils/logging.hpp"
    "${GPU_INCLUDE_DIR}/utils/array_utils.hpp"
    "${GPU_INCLUDE_DIR}/numeric_types/half.hpp"
)
    
add_library(gpu STATIC ${GPU_SOURCES} ${GPU_HEADERS})
set_target_properties(gpu PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(gpu PUBLIC "${GPU_INCLUDE_DIR}")
if(NOT EMSCRIPTEN)
    target_include_directories(gpu PUBLIC "${DAWN_BUILD_DIR}/gen/include/dawn/")
else()
    target_include_directories(gpu PUBLIC "${DAWN_BUILD_DIR}/gen/src/emdawnwebgpu/include/")
    target_include_directories(gpu PUBLIC "${DAWN_BUILD_DIR}/gen/src/emdawnwebgpu/include/webgpu/")
endif()
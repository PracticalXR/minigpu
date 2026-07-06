cmake_minimum_required(VERSION 3.22.1)
project(minigpu_ffi VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 20)

# Ensure iOS 13+ so std::filesystem APIs are available in simulator
if(IOS)
  set(MINIGPU_IOS_DEPLOYMENT_TARGET "16.0" CACHE STRING "Minimum iOS version" FORCE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "${MINIGPU_IOS_DEPLOYMENT_TARGET}" CACHE STRING "" FORCE)
  set(CMAKE_XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET "${MINIGPU_IOS_DEPLOYMENT_TARGET}" CACHE STRING "" FORCE)
  # Also set for the simulator explicitly
  set(CMAKE_XCODE_ATTRIBUTE_IPHONESIMULATOR_DEPLOYMENT_TARGET "${MINIGPU_IOS_DEPLOYMENT_TARGET}" CACHE STRING "" FORCE)
endif()

if(EMSCRIPTEN)
# Locate the Project Root
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/find_root.cmake")
find_project_root("${CMAKE_CURRENT_SOURCE_DIR}" "${FILENAME}" TARGET_FILE_PATH)
if(TARGET_FILE_PATH)
    message(STATUS "Project root found at ${TARGET_FILE_PATH}")
else()
    message(FATAL_ERROR "File ${FILENAME} not found within 8 levels up from ${CMAKE_CURRENT_SOURCE_DIR}")
endif()
set(WEB_OUTPUT_DIR ${TARGET_FILE_PATH}/minigpu_web/lib/web)
endif()

# Include Additional CMake Modules
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/dawn.cmake")

# Arch-aware output directories (reuse DAWN_ARCH from dawn.cmake)
if(NOT EMSCRIPTEN)
  if(EMSCRIPTEN)
    set(_mg_build_os "web")
  elseif(WIN32)
    set(_mg_build_os "win")
  elseif(IOS)
    set(_mg_build_os "ios")
  elseif(APPLE)
    set(_mg_build_os "mac")
  elseif(ANDROID)
    set(_mg_build_os "android")
  else()
    set(_mg_build_os "unix")
  endif()

  set(MINIGPU_OUT_ROOT "${CMAKE_BINARY_DIR}/out_${_mg_build_os}_${DAWN_ARCH}")
  message(STATUS "minigpu: target OS=${_mg_build_os}, arch=${DAWN_ARCH}, out=${MINIGPU_OUT_ROOT}")

  # Multi-config generators need per-config dirs as well
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${MINIGPU_OUT_ROOT}/lib")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${MINIGPU_OUT_ROOT}/lib")
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${MINIGPU_OUT_ROOT}/bin")
  foreach(cfg IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${cfg}" cfg_up)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${cfg_up} "${MINIGPU_OUT_ROOT}/lib")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${cfg_up} "${MINIGPU_OUT_ROOT}/lib")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg_up} "${MINIGPU_OUT_ROOT}/bin")
  endforeach()
endif()

# Define Main Target Names and Paths
if(NOT EMSCRIPTEN)
    set(MAIN_LIB minigpu_ffi)
else()
    set(MAIN_LIB minigpu_web)
    set(MAIN_JS  minigpu_web_js)
endif()

set(MAIN_PATH      ${CMAKE_CURRENT_SOURCE_DIR})
set(MAIN_OUT_PATH  ${CMAKE_CURRENT_BINARY_DIR})

#  Create the Main Target 
if(NOT EMSCRIPTEN)
    add_library(${MAIN_LIB} SHARED)
else()
    add_executable(${MAIN_LIB})
endif()

# Add Source Files and Include Directories
file(GLOB_RECURSE MAIN_SOURCES "${MAIN_PATH}/src/*.cpp")
target_sources(${MAIN_LIB} PRIVATE ${MAIN_SOURCES})
target_include_directories(${MAIN_LIB} PUBLIC ${MAIN_INCLUDES})
target_compile_definitions(${MAIN_LIB} PUBLIC DART_SHARED_LIB)

# ── Optional: fold EXTRA sources into the main target. Generic hook used to
# build single-blob wasm modules that link client codecs against minigpu
# in-module (e.g. the livetensor crosscoder, GSPLATS420_C_GPU_DESIGN.md §9).
# Configure with:
#   -DMGPU_EXTRA_SOURCES="<src1>;<src2>;..."
#   -DMGPU_EXTRA_INCLUDE_DIRS="<dir1>;..."
#   -DMGPU_EXTRA_DEFINES="<DEF1>;..."
# Extra sources should mark their exports EMSCRIPTEN_KEEPALIVE. ──
set(MGPU_EXTRA_SOURCES "" CACHE STRING "Extra sources compiled into the main target")
set(MGPU_EXTRA_INCLUDE_DIRS "" CACHE STRING "Include dirs for the extra sources")
set(MGPU_EXTRA_DEFINES "" CACHE STRING "Compile definitions for the extra sources")
if(MGPU_EXTRA_SOURCES)
    target_sources(${MAIN_LIB} PRIVATE ${MGPU_EXTRA_SOURCES})
    if(MGPU_EXTRA_INCLUDE_DIRS)
        target_include_directories(${MAIN_LIB} PRIVATE ${MGPU_EXTRA_INCLUDE_DIRS})
    endif()
    if(MGPU_EXTRA_DEFINES)
        target_compile_definitions(${MAIN_LIB} PRIVATE ${MGPU_EXTRA_DEFINES})
    endif()
endif()

# EMSCRIPTEN-Specific Settings
if(EMSCRIPTEN)
    # Include generated include directory before system includes
    include_directories(BEFORE "${DAWN_BUILD_DIR}/gen/src/emdawnwebgpu/include/")

    # Create a helper library for WebGPU (from Dawn)
    add_library(webgpu_web "${DAWN_DIR}/third_party/emdawnwebgpu/pkg/webgpu/src/webgpu.cpp")
    # Must pass --use-port at compile time so Emscripten installs the emdawnwebgpu
    # port headers before the sysroot webgpu.h (which is an older incompatible version).
    target_compile_options(webgpu_web PRIVATE
        "--use-port=${DAWN_DIR}/src/emdawnwebgpu/pkg/emdawnwebgpu-v20250818.200250.remoteport.py:shared_memory=true"
    )

    set(EMSCRIPTEN_DEBUG           OFF CACHE INTERNAL "Enable Emscripten debugging features" FORCE)

    # minigpu's own C ABI never crosses an asyncify operation through an
    # indirect call, so IGNORE_INDIRECT is a safe size/speed win. Extra
    # sources (single-blob clients) may use vtables on the asyncify path
    # (e.g. the crosscoder's backend dispatch around readSync) — IGNORE
    # would silently corrupt those frames, so drop it when they're present.
    if(MGPU_EXTRA_SOURCES)
        set(MGPU_ASYNCIFY_INDIRECT_FLAG "")
        message(STATUS "MGPU_EXTRA_SOURCES set: ASYNCIFY_IGNORE_INDIRECT disabled")
    else()
        set(MGPU_ASYNCIFY_INDIRECT_FLAG "-sASYNCIFY_IGNORE_INDIRECT=1")
    endif()

    message(STATUS "Needs Emscripten 4.0.10+, please upgrade if you face issues.")
    message(STATUS "toolchain dir ${EMSCRIPTEN_DIR}")
    if(EMSCRIPTEN_DEBUG)
        set(DEBUG_FLAGS "\
        -g3 \
        -O0 \
        -sSTACK_OVERFLOW_CHECK=2 \
        -sASSERTIONS=2 \
        -sDISABLE_EXCEPTION_CATCHING=0 \
        -sRUNTIME_DEBUG=1 \
        -sWARN_ON_UNDEFINED_SYMBOLS=1 \
        -sSAFE_HEAP=1 \
        -sNO_DISABLE_EXCEPTION_CATCHING \
        --source-map-base=http://localhost:8080/ \
    ")
    else()
        # Release flags
        set(DEBUG_FLAGS "\
            -O3 \
            -flto \
            -ffast-math \
            -DNDEBUG \
            -sMINIFY_HTML=1 \
            -sFILESYSTEM=0 \
            -sENVIRONMENT=web \
            -sMALLOC=emmalloc \
            -sNO_EXIT_RUNTIME=1 \
            -sTEXTDECODER=2 \
            -sIGNORE_MISSING_MAIN=1 \
        ")
    endif()

    # (Additional compile options could be added here)
    set_target_properties(${MAIN_LIB} PROPERTIES LINK_FLAGS "\
        ${DEBUG_FLAGS} \
        -sUSE_WEBGPU=0 \
        -sWASM=1 \
        -DDAWN_EMSCRIPTEN_TOOLCHAIN=${EMSCRIPTEN_DIR} \
        -sEXPORTED_FUNCTIONS=_malloc,_free,_memcpy \
        -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,HEAPU8,HEAPU16,HEAPU32,HEAP8,HEAP16,HEAP32,HEAPF32,HEAPF64 \
        -sUSE_GLFW=3 \
        -sINITIAL_MEMORY=256MB \
        -sSTACK_SIZE=64MB \
        -sALLOW_MEMORY_GROWTH=1 \
        -sMAXIMUM_MEMORY=2GB \
        -sASYNCIFY \
        -sASYNCIFY_STACK_SIZE=64MB \
        ${MGPU_ASYNCIFY_INDIRECT_FLAG} \
        -sEXIT_RUNTIME=0 \
        -sWASM_BIGINT=1 \
        --use-port=${DAWN_DIR}/src/emdawnwebgpu/pkg/emdawnwebgpu-v20250818.200250.remoteport.py:shared_memory=true \
        --closure-args=--externs=${EMSCRIPTEN_DIR}/src/closure-externs/webgpu-externs.js \
    ")
else()
    # Compiler-specific optimizations
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # Only use CPU-specific tuning when we are not on Apple and not cross-compiling
        set(CPU_TUNING_FLAGS "")
        if(NOT APPLE AND CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL CMAKE_SYSTEM_PROCESSOR)
            list(APPEND CPU_TUNING_FLAGS -march=native -mtune=native)
        endif()

        target_compile_options(${MAIN_LIB} PRIVATE 
            -Wall
            -Wno-error
            -ffast-math            # Aggressive floating-point optimizations
            -flto                  # Link-time optimization
            ${CPU_TUNING_FLAGS}
        )
        
        # Release-specific flags
        target_compile_options(${MAIN_LIB} PRIVATE 
            "$<$<NOT:$<CONFIG:Debug>>:-O3>"
            "$<$<NOT:$<CONFIG:Debug>>:-DNDEBUG>"
            "$<$<NOT:$<CONFIG:Debug>>:-fomit-frame-pointer>"
        )
        
        # Enable LTO for release builds
        set_target_properties(${MAIN_LIB} PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
            INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE
        )
    elseif(MSVC)
        target_compile_options(${MAIN_LIB} PRIVATE 
            /fp:fast               # Fast floating-point model
            /GL                    # Whole program optimization
        )
        
        # Release-specific flags
        target_compile_options(${MAIN_LIB} PRIVATE 
            "$<$<NOT:$<CONFIG:Debug>>:/O2>"
            "$<$<NOT:$<CONFIG:Debug>>:/Ob2>"    # Aggressive inlining
            "$<$<NOT:$<CONFIG:Debug>>:/DNDEBUG>"
        )
        
        # Link-time code generation for release
        set_target_properties(${MAIN_LIB} PROPERTIES
            LINK_FLAGS_RELEASE "/LTCG"
            LINK_FLAGS_RELWITHDEBINFO "/LTCG"
        )
    elseif(ANDROID)
    endif()
    
    # Additional optimizations
    target_compile_definitions(${MAIN_LIB} PRIVATE
        "$<$<NOT:$<CONFIG:Debug>>:NDEBUG>"
    )
    
endif()

if(NOT EMSCRIPTEN)
    target_include_directories(${MAIN_LIB} PUBLIC "${DAWN_BUILD_DIR}/gen/include/dawn/")
else()
    target_include_directories(${MAIN_LIB} PUBLIC "${DAWN_BUILD_DIR}/gen/src/emdawnwebgpu/include/")
    target_include_directories(${MAIN_LIB} PUBLIC "${DAWN_BUILD_DIR}/gen/src/emdawnwebgpu/include/webgpu/")
endif()

# Android Specific Linking
if(ANDROID)
     find_library(ANDROID_LOG_LIB log)
  if(ANDROID_LOG_LIB)
    target_link_libraries(${MAIN_LIB} PRIVATE ${ANDROID_LOG_LIB})
    target_link_libraries(webgpu_dawn PRIVATE ${ANDROID_LOG_LIB})
  else()
    message(FATAL_ERROR "Android log library not found")
  endif()
endif()

# Link Dawn monolithic lib; iOS links static archive
if(NOT EMSCRIPTEN)
  target_link_libraries(${MAIN_LIB} PRIVATE webgpu_dawn)
  if(IOS)
    target_link_libraries(${MAIN_LIB} PRIVATE
      "-framework Metal"
      "-framework QuartzCore"
      "-framework Foundation"
    )
  endif()
  # Platform-specific libs for minigpu_external.cpp
  if(WIN32)
    target_link_libraries(${MAIN_LIB} PRIVATE d3d11 d3d12 dxgi)
  endif()
  if(APPLE)
    target_link_libraries(${MAIN_LIB} PRIVATE
      "-framework IOSurface"
      "-framework Metal"
    )
  endif()
  if(ANDROID)
    find_library(ANDROID_HARDWARE_BUFFER_LIB android)
    target_link_libraries(${MAIN_LIB} PRIVATE ${ANDROID_HARDWARE_BUFFER_LIB})
  endif()
endif()

# Add Test Target (Non-Emscripten Only)
if(NOT EMSCRIPTEN)
    set(TEST_SOURCES
        "${MAIN_PATH}/test/minigpu_test.cpp"
        "${MAIN_PATH}/test/external_texture_test.cpp"
    )
    add_executable(minigpu_test ${TEST_SOURCES})
    # For tests, you may also want to expose the test sources in the main target;
    # adjust the following as needed
    target_link_libraries(minigpu_test PRIVATE ${MAIN_LIB})
    target_link_libraries(minigpu_test PRIVATE webgpu_dawn)
    if(WIN32)
        target_link_libraries(minigpu_test PRIVATE d3d11 d3d12 dxgi)
    endif()
    target_link_libraries(${MAIN_LIB} PRIVATE webgpu_dawn)
    # Dawn native include dirs needed for minigpu_external.cpp and test headers
    target_include_directories(${MAIN_LIB} PUBLIC
        "${MAIN_PATH}/include"
        "${DAWN_DIR}/include"
        "${DAWN_BUILD_DIR}/gen/include"
    )
    target_compile_definitions(${MAIN_LIB} PRIVATE DART_SHARED_LIB_TEST)
else()
    target_link_libraries(${MAIN_LIB} PRIVATE webgpu_web)
endif()

# EMSCRIPTEN: Copy JS Sources and Build Outputs to WEB_OUTPUT_DIR
if(EMSCRIPTEN)
    # 1. Copy additional JS sources (e.g. support libraries) from the build_web folder
    file(GLOB JS_SOURCES "${MAIN_PATH}/build_web/*.js")
    set(JS_OUTPUTS "")
    foreach(JS_FILE ${JS_SOURCES})
        get_filename_component(JS_FILENAME ${JS_FILE} NAME)
        set(JS_OUTPUT "${WEB_OUTPUT_DIR}/${JS_FILENAME}")
        list(APPEND JS_OUTPUTS ${JS_OUTPUT})
        add_custom_command(
            OUTPUT  ${JS_OUTPUT}
            COMMAND ${CMAKE_COMMAND} -E copy ${JS_FILE} ${JS_OUTPUT}
            DEPENDS ${JS_FILE}
            COMMENT "Copying ${JS_FILENAME} to web output directory"
        )
    endforeach()
    add_custom_target(${MAIN_JS} ALL
        DEPENDS ${JS_OUTPUTS}
        COMMENT "Copying JS support files to web output directory"
    )

    # 2. Copy the EMSCRIPTEN-built JS and WASM output files to WEB_OUTPUT_DIR.
    #    (We assume the output files are named after MAIN_LIB.)
    add_custom_command(TARGET ${MAIN_LIB} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_CURRENT_BINARY_DIR}/${MAIN_LIB}.js
                ${WEB_OUTPUT_DIR}/${MAIN_LIB}.js
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${CMAKE_CURRENT_BINARY_DIR}/${MAIN_LIB}.wasm
                ${WEB_OUTPUT_DIR}/${MAIN_LIB}.wasm
        COMMENT "Copying output JS and WASM to web output directory"
    )
endif()
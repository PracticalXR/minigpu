cmake_minimum_required(VERSION 3.22.1)
project(minigpu_ffi VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 20)

# Scoped capture sources size their shared_ptr control blocks from the VS 2022
# C++ library (include/minigpu_allocation_extent.h). Another toolset still
# builds the library, but those sources then report Unsupported at run time.
if(WIN32 AND (NOT MSVC OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
    NOT MSVC_TOOLSET_VERSION EQUAL 143 OR NOT CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "x64"))
  message(WARNING "minigpu_ffi: scoped capture sources need MSVC v143 (VS 2022) for x64; "
    "this ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} build reports them as unsupported.")
endif()

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
set(WEB_OUTPUT_DIR "${TARGET_FILE_PATH}/minigpu_web/lib/web" CACHE PATH
    "Destination for generated web artifacts")
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
    # Keep generated linker products out of the source-adjacent build root.
    # Besides avoiding stale-output locks on Windows, this prevents the JS
    # support-file glob below from mistaking the generated module for an input.
    set_target_properties(${MAIN_LIB} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated"
    )
endif()

# Add Source Files and Include Directories
# CONFIGURE_DEPENDS re-runs the glob at build time, so an added or deleted
# source reaches an existing build dir without a manual reconfigure.
file(GLOB_RECURSE MAIN_SOURCES CONFIGURE_DEPENDS "${MAIN_PATH}/src/*.cpp")
target_sources(${MAIN_LIB} PRIVATE ${MAIN_SOURCES})
target_include_directories(${MAIN_LIB} PUBLIC ${MAIN_INCLUDES})
target_compile_definitions(${MAIN_LIB} PUBLIC DART_SHARED_LIB)

# ── Dart dynamic-linking API (native builds only) ──────────────────────────
# third_party/dart_dl is a verbatim copy of the Dart SDK's include/ plus
# dart_api_dl.c. It lets mgpu::Logger deliver log lines to a Dart NATIVE PORT
# instead of an MGPULogCallback function pointer. The log registry is
# PROCESS-GLOBAL: a Dart NativeCallable registered there outlives the isolate
# that created it, and the VM aborts the whole process the moment a Dawn
# worker thread logs afterwards. A stale port id is inert.
#
# Not built for web — a wasm module has no Dart VM to post to, and the glob
# above only picks up .cpp so dart_api_dl.c is listed explicitly.
if(NOT EMSCRIPTEN)
    target_sources(${MAIN_LIB} PRIVATE
        "${MAIN_PATH}/third_party/dart_dl/dart_api_dl.c")
    target_include_directories(${MAIN_LIB} PRIVATE
        "${MAIN_PATH}/third_party/dart_dl")
    target_compile_definitions(${MAIN_LIB} PRIVATE MINIGPU_HAVE_DART_DL)
endif()

# ── Optional: fold EXTRA sources into the main target. Generic hook used to
# build single-blob wasm modules that link client codecs against minigpu
# in-module (e.g. a client video codec).
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

    # Resolve the emdawnwebgpu Emscripten port file.
    #
    # This must come from Dawn's ASSEMBLED package, not from Dawn's source tree:
    # the port file checks for generated headers copied in beside it and refuses
    # to run otherwise ("emdawnwebgpu.port.py must sit in a built
    # emdawnwebgpu_pkg"). Dawn's `emdawnwebgpu_pkg` custom target performs that
    # assembly into ${CMAKE_BINARY_DIR}/emdawnwebgpu_pkg, so the path below is
    # only populated after that target runs — hence the add_dependencies() and
    # deliberately NO EXISTS check on it (a clean tree has not built it yet).
    #
    # Older Dawn instead shipped a self-contained, version-stamped
    # `emdawnwebgpu-v<stamp>.remoteport.py` that fetched a released package, and
    # this file used to hardcode that name. Bumping DAWN_COMMIT therefore broke
    # the web build outright, and em++ only said "not a valid port path" ~370
    # targets in, long after Dawn itself had compiled. Both layouts now resolve.
    set(MGPU_EMDAWNWEBGPU_PORT
        "${CMAKE_BINARY_DIR}/emdawnwebgpu_pkg/emdawnwebgpu.port.py")
    if(NOT EXISTS "${DAWN_DIR}/src/emdawnwebgpu/pkg/emdawnwebgpu.port.py")
        # No in-tree port to assemble from: fall back to a stamped remote port,
        # which is standalone-usable, or stop with something readable.
        file(GLOB _mgpu_port_candidates
             "${DAWN_DIR}/src/emdawnwebgpu/pkg/emdawnwebgpu-v*.remoteport.py")
        if(_mgpu_port_candidates)
            list(GET _mgpu_port_candidates 0 MGPU_EMDAWNWEBGPU_PORT)
        else()
            message(FATAL_ERROR
                "emdawnwebgpu port file not found under "
                "${DAWN_DIR}/src/emdawnwebgpu/pkg\n"
                "Expected emdawnwebgpu.port.py, or a generated "
                "emdawnwebgpu-v*.remoteport.py.\n"
                "The Dawn checkout is probably at a commit whose layout differs "
                "— check DAWN_COMMIT in cmake/dawn.cmake.")
        endif()
    endif()
    message(STATUS "emdawnwebgpu port -> ${MGPU_EMDAWNWEBGPU_PORT}")

    # Create a helper library for WebGPU (from Dawn)
    add_library(webgpu_web "${DAWN_DIR}/third_party/emdawnwebgpu/pkg/webgpu/src/webgpu.cpp")
    # The port directory is assembled by this target; without the edge, ninja is
    # free to compile webgpu_web first and --use-port points at nothing.
    if(TARGET emdawnwebgpu_pkg)
        add_dependencies(webgpu_web emdawnwebgpu_pkg)
    endif()
    # Must pass --use-port at compile time so Emscripten installs the emdawnwebgpu
    # port headers before the sysroot webgpu.h (which is an older incompatible version).
    target_compile_options(webgpu_web PRIVATE
        "--use-port=${MGPU_EMDAWNWEBGPU_PORT}:shared_memory=true"
    )

    # Overridable with -DMGPU_WEB_DEBUG=ON: builds the wasm with -sASSERTIONS=2
    # etc. (below). 🔴 That is THE diagnostic for asyncify contract violations:
    # the release build fails them SILENTLY (an awaited async ccall resolves
    # while its C++ is still suspended — observed as init stopping after
    # "Requesting WebGPU adapter..." and readDirect "returning" zeros), while
    # the assertions build names the exact violation at the exact call.
    option(MGPU_WEB_DEBUG "Debug Emscripten build (assertions, -O0; slow, large)" OFF)
    set(EMSCRIPTEN_DEBUG ${MGPU_WEB_DEBUG})

    # 🔴 ASYNCIFY_IGNORE_INDIRECT is PERMANENTLY OFF (2026-08-17). The old
    # rationale ("minigpu's own C ABI never crosses an asyncify operation
    # through an indirect call") stopped being true: the emdawnwebgpu port's
    # event dispatch invokes completion callbacks (buffer-map, adapter and
    # device requests) THROUGH FUNCTION POINTERS while the caller is
    # suspended in a platformSleep poll loop. Under IGNORE_INDIRECT those
    # frames are uninstrumented and asyncify silently corrupts them — the
    # OBSERVED failure in the plain (no-extra-sources) build was an awaited
    # async ccall RESOLVING while its C++ never completed: init stopped
    # after "Requesting WebGPU adapter..." yet returned success, and every
    # readDirect logged its entry, suspended, and "returned" zeros. The
    # single-blob build (MGPU_EXTRA_SOURCES) never had the flag, which is
    # why the same wasm machinery worked there and failed here.
    set(MGPU_ASYNCIFY_INDIRECT_FLAG "")

    message(STATUS "Needs Emscripten 4.0.10+, please upgrade if you face issues.")
    message(STATUS "toolchain dir ${EMSCRIPTEN_DIR}")
    # C++ exception CATCHING stays OFF on web.
    # JS-based catching (-sDISABLE_EXCEPTION_CATCHING=0) routes every call that
    # may throw through a JS invoke_* trampoline, a tax on every call, and
    # native wasm exceptions (-fwasm-exceptions) do not work with -sASYNCIFY.
    # So a C++ throw aborts the module and every catch(...) fallback is dead
    # code on web: an exported function must check its preconditions BEFORE
    # anything that can throw instead of relying on a catch. The debug
    # variant's two catching flags are link-only, so minigpu's own objects
    # are compiled without catch support there too.
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
        # Release flags.
        #
        # 🔴 `-sENVIRONMENT=web,worker`, NOT `web`. `web` alone hard-codes
        # `ENVIRONMENT_IS_WORKER = false` in the generated glue and strips the
        # worker bootstrap, so the module cannot be instantiated inside a
        # dedicated Worker at all. That one token confined every GPU workload
        # on this platform to the main thread: WebGPU objects are not
        # transferable between threads, so a pipeline can only ever be used by
        # the thread that built it, and anything that must stay off the UI
        # thread therefore could not use this backend. Costs a little glue.
        #
        # 🔴 AND KEEP COMMENTS OUT OF THE STRING. These flags are one quoted,
        # backslash-continued CMake string: a `#` inside it is not a comment,
        # it is passed to the linker, and em++ answers with the memorable
        # `error: #: No such file or directory`.
        #
        # No -ffast-math: same reason as the native flags below (it lets the
        # optimizer drop the NaN/Inf input validation).
        set(DEBUG_FLAGS "\
            -O3 \
            -flto \
            -DNDEBUG \
            -sMINIFY_HTML=1 \
            -sFILESYSTEM=0 \
            -sENVIRONMENT=web,worker \
            -sMALLOC=emmalloc \
            -sNO_EXIT_RUNTIME=1 \
            -sTEXTDECODER=2 \
            -sIGNORE_MISSING_MAIN=1 \
        ")
    endif()

    # (Additional compile options could be added here)
    #
    # -sASYNCIFY_STACK_SIZE=1MB: the buffer a suspension unwinds the wasm call
    # stack into, malloc'd on EVERY sleep (each awaited read, poll-loop sleep,
    # init); 64MB grew the heap on the first read and pinned 64MB per call in
    # flight. Asyncify saves each frame's locals plus a call index, and the
    # largest function in the shipped module has 41 locals (<= ~330 bytes a
    # frame), so even a 100-frame-deep suspension needs ~33KB: 1MB is 30x that
    # (Emscripten's default is 4KB). Too small does not corrupt anything: the
    # unwind traps on `unreachable`.
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
        -sASYNCIFY_STACK_SIZE=1MB \
        ${MGPU_ASYNCIFY_INDIRECT_FLAG} \
        -sEXIT_RUNTIME=0 \
        -sWASM_BIGINT=1 \
        --use-port=${MGPU_EMDAWNWEBGPU_PORT}:shared_memory=true \
        --closure-args=--externs=${EMSCRIPTEN_DIR}/src/closure-externs/webgpu-externs.js \
        --post-js ${CMAKE_CURRENT_SOURCE_DIR}/web/webgpu_adapter_info_guard.post.js \
    ")

    # Emscripten finalizes JavaScript through Python's shutil.move(). On
    # Windows that operation does not replace an existing destination, so an
    # otherwise successful incremental link stalls/fails at final emission.
    # Remove only this build target's stale intermediates immediately before a
    # relink. The separately packaged web runtime remains last-good until the
    # post-build copy runs after a successful link.
    add_custom_command(TARGET ${MAIN_LIB} PRE_LINK
        COMMAND "${CMAKE_COMMAND}" -E rm -f
            "$<TARGET_FILE:${MAIN_LIB}>"
            "$<TARGET_FILE_DIR:${MAIN_LIB}>/${MAIN_LIB}.wasm"
        VERBATIM
    )
else()
    # Compiler-specific optimizations
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # No -march=native: this library ships inside apps, so the build
        # machine's ISA (AVX2/AVX-512) would SIGILL on older user CPUs.
        # No -ffast-math (nor /fp:fast below): it implies finite-math-only,
        # which folds the library's std::isfinite / NaN input validation
        # (sampler LOD clamps, clear values) to "valid". This is handle and
        # validation code, not numeric kernels.
        target_compile_options(${MAIN_LIB} PRIVATE
            -Wall
            -Wno-error
            -flto                  # Link-time optimization
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
    add_executable(minigpu_buffer_descriptor_test
        "${MAIN_PATH}/test/buffer_descriptor_test.cpp")
    target_link_libraries(minigpu_buffer_descriptor_test PRIVATE ${MAIN_LIB})
    target_link_libraries(minigpu_buffer_descriptor_test PRIVATE webgpu_dawn)
    add_executable(minigpu_context_lifetime_test
        "${MAIN_PATH}/test/context_lifetime_test.cpp")
    target_link_libraries(minigpu_context_lifetime_test PRIVATE ${MAIN_LIB})
    target_link_libraries(minigpu_context_lifetime_test PRIVATE webgpu_dawn)
    add_executable(minigpu_queue_completion_boundary_test
        "${MAIN_PATH}/test/queue_completion_boundary_test.cpp")
    target_link_libraries(minigpu_queue_completion_boundary_test PRIVATE
        ${MAIN_LIB} webgpu_dawn)
    add_executable(minigpu_shader_exclusive_boundary_test
        "${MAIN_PATH}/test/shader_exclusive_boundary_test.cpp")
    target_link_libraries(minigpu_shader_exclusive_boundary_test PRIVATE
        ${MAIN_LIB} webgpu_dawn)
    add_executable(minigpu_imported_frame_boundary_test
        "${MAIN_PATH}/test/imported_frame_boundary_test.cpp"
        "${MAIN_PATH}/test/imported_frame_abi_c.c")
    target_link_libraries(minigpu_imported_frame_boundary_test PRIVATE
        ${MAIN_LIB} webgpu_dawn)
    add_executable(minigpu_resource_cleanup_boundary_test
        "${MAIN_PATH}/test/resource_cleanup_boundary_test.cpp"
        "${MAIN_PATH}/test/resource_cleanup_abi_c.c")
    target_link_libraries(minigpu_resource_cleanup_boundary_test PRIVATE
        ${MAIN_LIB} webgpu_dawn)
    add_executable(minigpu_private_output_boundary_test
        "${MAIN_PATH}/test/private_output_boundary_test.cpp"
        "${MAIN_PATH}/test/private_output_abi_c.c")
    target_link_libraries(minigpu_private_output_boundary_test PRIVATE
        ${MAIN_LIB} webgpu_dawn)
    add_executable(minigpu_completion_ticket_cold_test
        "${MAIN_PATH}/test/completion_ticket/completion_ticket_cold_test.cpp"
        "${MAIN_PATH}/test/completion_ticket/completion_ticket_abi_c.c")
    target_link_libraries(minigpu_completion_ticket_cold_test PRIVATE
        ${MAIN_LIB} webgpu_dawn)
    add_executable(minigpu_private_resources_cold_test
        "${MAIN_PATH}/test/private_resources_cold_test.cpp"
        "${MAIN_PATH}/test/private_resources_abi_c.c")
    target_link_libraries(minigpu_private_resources_cold_test PRIVATE
        ${MAIN_LIB} webgpu_dawn)
    add_executable(minigpu_capture_source_cold_test
        "${MAIN_PATH}/test/capture_source/capture_source_cold_test.cpp"
        "${MAIN_PATH}/test/capture_source/capture_source_abi_c.c")
    target_link_libraries(minigpu_capture_source_cold_test PRIVATE
        ${MAIN_LIB} webgpu_dawn)
    add_executable(minigpu_external_pool_test
        "${MAIN_PATH}/test/external_pool_test.cpp")
    target_link_libraries(minigpu_external_pool_test PRIVATE ${MAIN_LIB} webgpu_dawn)
    if(WIN32)
        target_link_libraries(minigpu_external_pool_test PRIVATE d3d11 dxgi)
    endif()
    add_executable(minigpu_texture_resource_test
        "${MAIN_PATH}/test/texture_resource_test.cpp")
    target_link_libraries(minigpu_texture_resource_test PRIVATE ${MAIN_LIB})
    target_link_libraries(minigpu_texture_resource_test PRIVATE webgpu_dawn)
    # The contract tests bring d3d11/dxgi in with #pragma comment(lib).
    add_executable(minigpu_external_contract_test
        "${MAIN_PATH}/test/external_contract_test.cpp")
    target_link_libraries(minigpu_external_contract_test PRIVATE
        ${MAIN_LIB} webgpu_dawn)
    add_executable(minigpu_render_contract_test
        "${MAIN_PATH}/test/render_contract_test.cpp")
    target_link_libraries(minigpu_render_contract_test PRIVATE
        ${MAIN_LIB} webgpu_dawn)
    if(WIN32)
        add_executable(minigpu_external_pool_contract_test
            "${MAIN_PATH}/test/external_pool_contract_test.cpp")
        target_link_libraries(minigpu_external_pool_contract_test PRIVATE
            ${MAIN_LIB} webgpu_dawn)
    endif()
    add_executable(minigpu_export_failure_test
        "${MAIN_PATH}/test/export_failure_test.cpp")
    target_link_libraries(minigpu_export_failure_test PRIVATE ${MAIN_LIB})
    target_link_libraries(minigpu_export_failure_test PRIVATE webgpu_dawn)
    add_executable(minigpu_render_resource_test
        "${MAIN_PATH}/test/render_resource_test.cpp")
    target_link_libraries(minigpu_render_resource_test PRIVATE ${MAIN_LIB})
    target_link_libraries(minigpu_render_resource_test PRIVATE webgpu_dawn)
    add_executable(minigpu_compute_guard_test
        "${MAIN_PATH}/test/compute_guard_test.cpp")
    target_link_libraries(minigpu_compute_guard_test PRIVATE
        ${MAIN_LIB} webgpu_dawn)
    add_executable(minigpu_shader_cache_files_test
        "${MAIN_PATH}/test/shader_cache_files_test.cpp")
    target_link_libraries(minigpu_shader_cache_files_test PRIVATE
        ${MAIN_LIB} webgpu_dawn)
    add_executable(minigpu_immutable_import_test
        "${MAIN_PATH}/test/immutable_frame_import_test.cpp")
    target_link_libraries(minigpu_immutable_import_test PRIVATE ${MAIN_LIB})
    add_executable(minigpu_compute_pipeline_async_test
        "${MAIN_PATH}/test/compute_pipeline_async_test.cpp")
    target_link_libraries(minigpu_compute_pipeline_async_test PRIVATE
        ${MAIN_LIB} webgpu_dawn)
    add_executable(minigpu_external_async_test
        "${MAIN_PATH}/test/external_async_test.cpp")
    target_link_libraries(minigpu_external_async_test PRIVATE
        ${MAIN_LIB} webgpu_dawn)
    if(WIN32)
        target_link_libraries(minigpu_test PRIVATE d3d11 d3d12 dxgi)
        target_link_libraries(minigpu_immutable_import_test PRIVATE d3d11 dxgi)
        target_link_libraries(minigpu_external_async_test PRIVATE d3d11 d3d12 dxgi)
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
    list(FILTER JS_SOURCES EXCLUDE REGEX "/${MAIN_LIB}\\.js$")
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
                $<TARGET_FILE:${MAIN_LIB}>
                ${WEB_OUTPUT_DIR}/${MAIN_LIB}.js
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE_DIR:${MAIN_LIB}>/${MAIN_LIB}.wasm
                ${WEB_OUTPUT_DIR}/${MAIN_LIB}.wasm
        COMMENT "Copying output JS and WASM to web output directory"
    )
endif()

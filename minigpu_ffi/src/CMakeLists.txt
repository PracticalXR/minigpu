cmake_minimum_required(VERSION 3.5)
project(minigpu_ffi VERSION 1.0.0)
cmake_minimum_required(VERSION 3.22.1)
set(CMAKE_CXX_STANDARD 20)

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
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/gpu.cmake")

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

# EMSCRIPTEN-Specific Settings
if(EMSCRIPTEN)
    # Include generated include directory before system includes
    include_directories(BEFORE "${DAWN_BUILD_DIR}/gen/src/emdawnwebgpu/include/")

    # Create a helper library for WebGPU (from Dawn)
    add_library(webgpu_web "${DAWN_DIR}/third_party/emdawnwebgpu/webgpu.cpp")

    set(EMSCRIPTEN_DEBUG           OFF CACHE INTERNAL "Enable Emscripten debugging features" FORCE)

    message(STATUS "toolchain dir ${EMSCRIPTEN_DIR}")
    if(EMSCRIPTEN_DEBUG)
        set(DEBUG_FLAGS "\
            -g3 \
            -O0 \
            -sSTACK_OVERFLOW_CHECK=2 \
            -sRUNTIME_DEBUG=1 \
            -sWARN_ON_UNDEFINED_SYMBOLS=1 \
            -sASSERTIONS=2 \
            -fsanitize=address \
            -sNO_DISABLE_EXCEPTION_CATCHING \
            --source-map-base=http://localhost:8080/ \
        ")
    else()
        set(DEBUG_FLAGS "-O3")
    endif()

    # (Additional compile options could be added here)
    set_target_properties(${MAIN_LIB} PROPERTIES LINK_FLAGS "\
        ${DEBUG_FLAGS} \
        -sUSE_WEBGPU=0 \
        -sWASM=1 \
        -DDAWN_EMSCRIPTEN_TOOLCHAIN=${EMSCRIPTEN_DIR} \
        -sEXPORTED_FUNCTIONS=_malloc,_free,_memcpy \
        -sEXPORTED_RUNTIME_METHODS=ccall \
        -sUSE_GLFW=3 \
        -sINITIAL_MEMORY=16MB \
        -sMAXIMUM_MEMORY=2GB \
        -sALLOW_MEMORY_GROWTH=1 \
        -sStackSize=16MB \
        -sASYNCIFY \
        -sASYNCIFY_STACK_SIZE=262144 \
        -sASYNCIFY_IGNORE_INDIRECT=1 \
        -sRUNTIME_DEBUG=0 \
        -sEXIT_RUNTIME=0 \
        --js-library=${DAWN_BUILD_DIR}/gen/src/emdawnwebgpu/library_webgpu_enum_tables.js \
        --js-library=${DAWN_BUILD_DIR}/gen/src/emdawnwebgpu/library_webgpu_generated_struct_info.js \
        --js-library=${DAWN_BUILD_DIR}/gen/src/emdawnwebgpu/library_webgpu_generated_sig_info.js \
        --js-library=${DAWN_DIR}/third_party/emdawnwebgpu/library_webgpu.js \
        --closure-args=--externs=${EMSCRIPTEN_DIR}/src/closure-externs/webgpu-externs.js \
    ")
endif()

# Add Test Target (Non-Emscripten Only)
if(NOT EMSCRIPTEN)
    set(TEST_SOURCES "${MAIN_PATH}/test/minigpu_test.cpp")
    add_executable(minigpu_test ${TEST_SOURCES})
    # For tests, you may also want to expose the test sources in the main target;
    # adjust the following as needed
    target_link_libraries(minigpu_test PRIVATE gpu)
    target_link_libraries(minigpu_test PRIVATE ${MAIN_LIB})
    target_link_libraries(minigpu_test PRIVATE webgpu_dawn)
    target_link_libraries(gpu PRIVATE webgpu_dawn)
    target_link_libraries(${MAIN_LIB} PRIVATE webgpu_dawn)
else()
    target_link_libraries(${MAIN_LIB} PRIVATE webgpu_web)
endif()


# Link Libraries 

target_link_libraries(${MAIN_LIB} PRIVATE gpu)


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

# Android Specific Linking
if(ANDROID)
   # target_link_libraries(${MAIN_LIB} OpenSLES)
endif()
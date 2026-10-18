cmake_minimum_required(VERSION 3.14)
project(minigpu_capture_source_cpu LANGUAGES CXX)
find_package(Threads REQUIRED)
add_executable(minigpu_capture_source_cpu
  "../../src/minigpu_source_scope.cpp"
  "../../src/minigpu_capture_source_owner.cpp"
  "capture_source_cpu_test.cpp")
target_compile_features(minigpu_capture_source_cpu PRIVATE cxx_std_20)
target_link_libraries(minigpu_capture_source_cpu PRIVATE Threads::Threads)
if(MSVC)
  target_compile_options(minigpu_capture_source_cpu PRIVATE /W4 /WX /EHsc)
else()
  target_compile_options(minigpu_capture_source_cpu PRIVATE -Wall -Wextra -Werror)
endif()

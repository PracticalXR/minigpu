cmake_minimum_required(VERSION 3.14)
project(minigpu_capture_work_cpu LANGUAGES CXX)
add_executable(minigpu_capture_work_cpu
  "../../src/minigpu_source_scope.cpp"
  "../../src/minigpu_capture_recipe_owner.cpp"
  "../../src/minigpu_capture_cleanup_owner.cpp"
  "../../src/minigpu_capture_work_owner.cpp"
  "capture_work_cpu_test.cpp"
  "queue_inline_cpu_test.cpp")
target_compile_features(minigpu_capture_work_cpu PRIVATE cxx_std_17)
target_compile_definitions(minigpu_capture_work_cpu PRIVATE MINIGPU_INLINE_RESULT_TEST)
if(MSVC)
  target_compile_options(minigpu_capture_work_cpu PRIVATE /W4 /WX /EHsc)
else()
  target_compile_options(minigpu_capture_work_cpu PRIVATE -Wall -Wextra -Werror)
endif()

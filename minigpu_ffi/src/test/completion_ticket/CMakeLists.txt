cmake_minimum_required(VERSION 3.14)
project(minigpu_completion_ticket_cpu LANGUAGES C CXX)
add_executable(minigpu_completion_ticket_cpu
  "../../src/minigpu_completion_ticket.cpp" "completion_ticket_cpu_test.cpp"
  "../../src/minigpu_source_scope.cpp" "../../src/minigpu_capture_cleanup_owner.cpp"
  "../../src/minigpu_capture_work_owner.cpp" "../../src/minigpu_capture_recipe_owner.cpp"
  "completion_ticket_abi_c.c")
target_compile_features(minigpu_completion_ticket_cpu PRIVATE cxx_std_17)
target_compile_definitions(minigpu_completion_ticket_cpu PRIVATE
  MINIGPU_COMPLETION_TEST_PROXY_TAIL MINIGPU_SOURCE_METADATA_TEST)
if(MSVC)
  target_compile_options(minigpu_completion_ticket_cpu PRIVATE /W4 /WX /EHsc)
endif()

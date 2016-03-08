cmake_minimum_required(VERSION 3.2)
find_package(GFlags REQUIRED)
find_package(GTest REQUIRED)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -march=native -mtune=native -Wunused-parameter -Werror -Wno-char-subscripts -std=c++14 -O3 -g")

# auto vectorization report -qopt-report-phase=vec -qopt-report=5 -qopt-report-file=stdout

add_executable(runbench bench.cc impl.cc)
target_link_libraries(runbench gflags benchmark pthread)

add_executable(runtests tests.cc impl.cc)
target_link_libraries(runtests gtest pthread)
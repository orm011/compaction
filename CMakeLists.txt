cmake_minimum_required(VERSION 3.2)
find_package(GFlags REQUIRED)
find_package(GTest REQUIRED)


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -wd780 -march=native -mtune=native -Wunused-parameter -Wno-char-subscripts -std=c++14 -O3 -g")

# auto vectorization report -qopt-report-phase=vec -qopt-report=5 -qopt-report-file=stdout

add_executable(runbench runbench.cc impl.cc)
target_link_libraries(runbench gflags benchmark pthread gtest tbb)

add_executable(runtests runtests.cc impl.cc)
target_link_libraries(runtests gtest pthread tbb)
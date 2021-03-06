cmake_minimum_required(VERSION 3.2)
find_package(GFlags REQUIRED)
find_package(GTest REQUIRED)

INCLUDE_DIRECTORIES(/home/orm/IntelPerformanceCounterMonitor-PCM-V2.10)
LINK_DIRECTORIES(/home/orm/IntelPerformanceCounterMonitor-PCM-V2.10/intelpcm.so)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -march=native -mtune=native -Wunused-parameter -Wno-char-subscripts -std=c++14 -O3 -g")

SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -debug")

if (DEFINED ENV{DATA_T})
	message("setting DATA_T to "  $ENV{DATA_T} "...")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDATA_T=$ENV{DATA_T}")
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -wd780 ")
endif()

# auto vectorization report -qopt-report-phase=vec -qopt-report=5 -qopt-report-file=stdout

add_executable(runbench runbench.cc impl.cc)
target_link_libraries(runbench  benchmark pthread gtest tbb gflags intelpcm)

add_executable(runtests runtests.cc impl.cc)
target_link_libraries(runtests gtest pthread tbb gflags)
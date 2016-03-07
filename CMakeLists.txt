find_package(gflags REQUIRED)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-parameter -Wno-unused-variable -Werror -Wno-char-subscripts -Wno-deprecated-declarations")

add_executable(main main.cc)
target_link_libraries(main gflags)
target_link_libraries(main benchmark)
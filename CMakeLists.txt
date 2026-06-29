cmake_minimum_required(VERSION 3.10)
project(ACORN VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable -Wno-sign-compare -O3 -mavx2 -mfma")

# OpenMP
find_package(OpenMP REQUIRED)

# ACORN library
set(ACORN_SOURCES
    src/acorn_graph.cpp
    src/distances.cpp
    src/file_io.cpp
)

set(ACORN_HEADERS
    include/acorn/types.h
    include/acorn/distance.h
    include/acorn/acorn_graph.h
    include/acorn/file_io.h
)

add_library(acorn ${ACORN_SOURCES} ${ACORN_HEADERS})

target_include_directories(acorn
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(acorn PUBLIC OpenMP::OpenMP_CXX)

# Examples
add_executable(gen_labels examples/gen_labels.cpp)
target_link_libraries(gen_labels acorn)

add_executable(compute_filtered_gt examples/compute_filtered_gt.cpp)
target_link_libraries(compute_filtered_gt acorn)

add_executable(search examples/search.cpp)
target_link_libraries(search acorn)

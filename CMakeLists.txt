cmake_minimum_required(VERSION 3.10)
project(ACORN VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable -Wno-sign-compare -O3 -mavx2 -mfma")

# OpenMP
find_package(OpenMP REQUIRED)

# ACORN library
set(ACORN_SOURCES
    src/index.cpp
    src/acorn_graph.cpp
    src/index_acorn.cpp
    src/random.cpp
    src/distances.cpp
    src/file_io.cpp
)

set(ACORN_HEADERS
    include/acorn/types.h
    include/acorn/visited.h
    include/acorn/distance.h
    include/acorn/heap.h
    include/acorn/random.h
    include/acorn/index.h
    include/acorn/acorn_graph.h
    include/acorn/index_acorn.h
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
add_executable(example examples/simple_example.cpp)
target_link_libraries(example acorn)

add_executable(eval_sift1m examples/eval_sift1m.cpp)
target_link_libraries(eval_sift1m acorn)

add_executable(gen_labels examples/gen_labels.cpp)
target_link_libraries(gen_labels acorn)

add_executable(compute_groundtruth examples/compute_groundtruth.cpp)
target_link_libraries(compute_groundtruth acorn)

add_executable(test_parallel examples/test_parallel.cpp)
target_link_libraries(test_parallel acorn)

add_executable(compute_filtered_gt examples/compute_filtered_gt.cpp)
target_link_libraries(compute_filtered_gt acorn)

# Optionally build tests
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    # add_subdirectory(tests)
endif()

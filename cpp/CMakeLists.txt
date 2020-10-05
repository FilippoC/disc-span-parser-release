cmake_minimum_required(VERSION 3.10)
project(disc_span_parser)

if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt.")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_CXX_STANDARD 11)

find_package(Boost COMPONENTS thread REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

# Python will only check for .so files, not .dylib files
if(APPLE)
    set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
endif(APPLE)

# Compilation flag
# May not be compatible with clang?
add_definitions("-Wall")
add_definitions("-Wextra")
add_definitions("-pedantic")
add_definitions("-O3")
add_definitions("-Wfatal-errors")
add_definitions("-funroll-loops")
add_definitions("-march=native")
add_definitions("-fPIC")

set(CMAKE_SHARED_LIBRARY_PREFIX "")

find_package(Python3 REQUIRED COMPONENTS Development)
message("Python lib: ${Python3_LIBRARIES}")
message("Python include dirs: ${Python3_INCLUDE_DIRS}")

include_directories(${Python3_INCLUDE_DIRS})

add_library(disc_pstruct SHARED
        cpp_disc_span_parser.cpp
        disc_pstruct/binarize.cpp
        disc_pstruct/corpus.cpp
        disc_pstruct/set.cpp
        disc_pstruct/tree.cpp
        disc_pstruct/argmax-disc/chart.cpp
        disc_pstruct/inside-outside-disc/chart.cpp
        disc_pstruct/data.h disc_pstruct/data.cpp disc_pstruct/argmax-disc/algorithms.cpp disc_pstruct/inside-outside-disc/algorithms.cpp test_pool.cpp cxxtimer.h)
target_link_libraries(disc_pstruct ${Python3_LIBRARIES})
target_link_libraries(disc_pstruct ${Boost_LIBRARIES})

add_executable(test_pool test_pool.cpp)
target_link_libraries(test_pool ${Boost_LIBRARIES})

add_executable(compute_max_recall compute_max_recall.cpp)
target_link_libraries(compute_max_recall disc_pstruct)
target_link_libraries(compute_max_recall ${Boost_LIBRARIES})

add_executable(filter_dataset filter_dataset.cpp)
target_link_libraries(filter_dataset disc_pstruct)
target_link_libraries(filter_dataset ${Boost_LIBRARIES})


cmake_minimum_required(VERSION 2.8.12)

project(OCVCascadeSampleGenerator)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} include)

set(SOURCES src/main.cpp)
set(CMAKE_BINARY_DIR bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

add_executable(OCVCascadeSampleGenerator src/main.cpp)
target_link_libraries(OCVCascadeSampleGenerator ${OpenCV_LIBS})
target_compile_options(OCVCascadeSampleGenerator PUBLIC -std=c++11 -Wall -Wfloat-conversion)

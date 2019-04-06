cmake_minimum_required(VERSION 3.9.5)
project(tracker)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3")

set(PANO_SOURCES
        pano.cpp)

set(FILTER_SOURCES
        filter.cpp)

find_package(OpenCV REQUIRED)
add_executable(tracker ${PANO_SOURCES})
add_executable(filter ${FILTER_SOURCES})

target_link_libraries(tracker ${OpenCV_LIBS})
target_link_libraries(filter ${OpenCV_LIBS})

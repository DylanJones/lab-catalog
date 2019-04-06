cmake_minimum_required(VERSION 3.9.5)
project(tracker)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3")

set(SOURCE_FILES
        tracker.cpp)

find_package(OpenCV REQUIRED)
add_executable(tracker ${SOURCE_FILES})
target_link_libraries(tracker ${OpenCV_LIBS})

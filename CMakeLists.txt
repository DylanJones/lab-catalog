cmake_minimum_required(VERSION 3.9.5)
project(tracker)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O0 -ggdb")

set(PANO_SOURCES
        pano.cpp)

set(PANO2_SOURCES
        pano2.cpp)

set(PANO_ORB_SOURCES
        pano_orb.cpp)

set(STITCHER_SOURCES
        pano_stitch.cpp)

set(FILTER_SOURCES
        filter.cpp)

find_package(OpenCV REQUIRED)
add_executable(pano ${PANO_SOURCES})
add_executable(pano2 ${PANO2_SOURCES})
add_executable(pano_orb ${PANO_ORB_SOURCES})
add_executable(filter ${FILTER_SOURCES})
add_executable(stitcher ${STITCHER_SOURCES})

target_link_libraries(pano ${OpenCV_LIBS})
target_link_libraries(pano2 ${OpenCV_LIBS})
target_link_libraries(pano_orb ${OpenCV_LIBS})
target_link_libraries(filter ${OpenCV_LIBS})
target_link_libraries(stitcher ${OpenCV_LIBS})

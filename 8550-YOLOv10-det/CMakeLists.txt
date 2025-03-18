cmake_minimum_required(VERSION 3.10)
project(YOLOv10s)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")

set(CURRENT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(OpenCV_DIR "${CURRENT_DIR}/3rd/opencv") 

# 查找OpenCV库
find_library(OpenCV_CORE_LIB opencv_core HINTS ${OpenCV_DIR}/lib)
find_library(OpenCV_IMGPROC_LIB opencv_imgproc HINTS ${OpenCV_DIR}/lib)
find_library(OpenCV_IMGCODECS_LIB opencv_imgcodecs HINTS ${OpenCV_DIR}/lib)

# 检查库是否找到
if(NOT OpenCV_CORE_LIB OR NOT OpenCV_IMGPROC_LIB OR NOT OpenCV_IMGCODECS_LIB)
    message(FATAL_ERROR "OpenCV libraries not found!")
endif()

set(OpenCV_INCLUDE_DIRS 
    "${OpenCV_DIR}/include/opencv4"
)

add_executable(test
    ./main.cpp
    ./SNPETask.cpp
    ./YOLOv10s.cpp
)

target_include_directories(test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    /usr/include/SNPE
)

target_link_libraries(test
    PRIVATE 
    pthread
    dl
    ${OpenCV_CORE_LIB}
    ${OpenCV_IMGPROC_LIB}
    ${OpenCV_IMGCODECS_LIB}
    /usr/lib/libSNPE.so
    ${CMAKE_DL_LIBS}
    -lz
)

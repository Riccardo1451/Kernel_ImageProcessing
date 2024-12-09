cmake_minimum_required(VERSION 3.28)
project(Kernel_Code)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})



set(CMAKE_CXX_STANDARD 17)

add_executable(Kernel_Code main.cpp
        kernel.cpp
        image.cpp
        image.h)

target_link_libraries(Kernel_Code ${OpenCV_LIBS})
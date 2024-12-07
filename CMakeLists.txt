cmake_minimum_required(VERSION 3.28)
project(Kernel_Code)

set(CMAKE_CXX_STANDARD 17)

add_executable(Kernel_Code main.cpp
        kernel.cpp)

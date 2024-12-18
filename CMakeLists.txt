cmake_minimum_required(VERSION 3.28)
project(Kernel_Code)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})



set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -lomp")

include_directories(/opt/homebrew/Cellar/libomp/19.1.2/include)
link_directories(/opt/homebrew/Cellar/libomp/19.1.2/lib)

add_executable(Kernel_Code main.cpp
        kernel.cpp
        image.cu)

target_link_libraries(Kernel_Code ${OpenCV_LIBS})
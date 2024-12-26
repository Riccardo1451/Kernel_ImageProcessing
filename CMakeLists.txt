cmake_minimum_required(VERSION 3.28)
project(Kernel_Code)

include_directories(${CMAKE_SOURCE_DIR}/include)

#Enable CUDA
enable_language(CUDA)

#CUDA Path
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.8)
#CUDA Compiler nvcc
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.8/bin/nvcc)
#CUDA Architecture
set(CMAKE_CUDA_ARCHITECTURES 75 86)



include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

#OpenCV
set(OpenCV_DIR "/data01/pc24ricfan/Desktop/opencv/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


find_package(OpenMP REQUIRED)
set(OPENMP_DIR "/data01/pc24ricfan/Desktop/openmp")
set(OpenMP_C_FLAGS "-fopenmp")
set(OpenMP_CXX_FLAGS "-fopenmp")
set(OpenMP_C_LIB_NAMES "omp")
set(OpenMP_CXX_LIB_NAMES "omp")
set(OpenMP_omp_LIBRARY "/data01/pc24ricfan/Desktop/openmp/lib/libomp.so") # o libomp.a
include_directories(${OPENMP_DIR}/include)
link_directories(${OPENMP_DIR}/lib)
set(CMAKE_BUILD_RPATH "${OPENMP_DIR}/lib")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")


set(CMAKE_CXX_STANDARD 17)

add_executable(Kernel_Code main.cpp
        kernel.cpp
        image.cpp
        CUDA_convolutions.cu
        CUDA_convolutions.cuh)

target_link_libraries(Kernel_Code
            cuda
            cudart
            ${OpenMP_CXX_LIBRARIES}
            ${OpenCV_LIBS})

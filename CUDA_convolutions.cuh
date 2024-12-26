#ifndef KERNEL_CODE_CUDA_CONVOLUTIONS_CUH
#define KERNEL_CODE_CUDA_CONVOLUTIONS_CUH

#include "kernel.h"
//Using global memory both for the image and the kernel vector
void applyGlobalCUDAConvolution(std::vector<float>& imageMatrix, const kernel& kernel, std::vector<float>& output, int imageWidth, int imageHeight);

//Using global memory for the image and constant memory for the kernel vector
void applyConstantCUDAConvolution(std::vector<float>& imageMatrix, const kernel& kernel, std::vector<float>& output, int imageWidth, int imageHeight);

//Using both constant and shared memory for kernel and image vector
void applySharedCUDAConvolution(std::vector<float>& imageMatrix, const kernel& kernel, std::vector<float>& output, int imageWidth, int imageHeight);
#endif //KERNEL_CODE_CUDA_CONVOLUTIONS_CUH

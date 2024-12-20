#ifndef KERNEL_CODE_CUDA_CONVOLUTIONS_CUH
#define KERNEL_CODE_CUDA_CONVOLUTIONS_CUH

#include "kernel.h"

class CUDA_convolutions {
public:
    //CUDA function wrapper
    void applyCUDAConvolution(std::vector<float>& imageMatrix, const kernel& kernel, std::vector<float>& output, int imageWidth, int imageHeight);
};


#endif //KERNEL_CODE_CUDA_CONVOLUTIONS_CUH

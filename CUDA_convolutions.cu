#include "CUDA_convolutions.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>


__global__ void applyCUDAConvolutionKernel(const float * ker,const float* mat, float* output, int width, int height, int kernelSize) {

    //Calculate global position for each thread
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    //We are already iterating thorugh the image, just with the x,y positions
    float sum = 0.0;
    int midKernel = kernelSize / 2;
    if(x < width && y < height) {
        for(int ki = 0; ki < midKernel; ki++) {
            for(int kj = 0; kj < midKernel; kj++) {

                float imgValue = mat[(y + ki + midKernel) * (width + 2 * midKernel) + (x + kj + midKernel)];
                float kernelValue = ker[(ki+midKernel)*kernelSize + (kj+midKernel)];

                sum += imgValue * kernelValue;
            }
        }
        output[y * width + x] = sum;
    }

}
void applyCUDAConvolution(std::vector<float>& imageMatrix, const kernel& kernel, std::vector<float>& output, int imageWidth, int imageHeight){
    //requiresPadding
    std::chrono::time_point<std::chrono::system_clock> start, end;

    //Set up grid and blocks
    dim3 blockDim(16,16);
    dim3 gridDim((imageWidth + blockDim.x - 1) / blockDim.x, (imageHeight + blockDim.y - 1) / blockDim.y);

    //Move data to GPU
    float *d_image, *d_kernel, *d_output;
    cudaMalloc(&d_image, imageMatrix.size()*sizeof(float));
    cudaMalloc(&d_kernel, kernel.getKernel().size()*sizeof(float));
    cudaMalloc(&d_output, output.size()*sizeof(float));

    cudaMemcpy(&d_image, imageMatrix.data(),imageMatrix.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_kernel, kernel.getKernel().data(),kernel.getKernel().size() * sizeof(float), cudaMemcpyHostToDevice);

    //Call kernel func
    start = std::chrono::system_clock::now();
    applyCUDAConvolutionKernel<<<gridDim, blockDim>>>(d_kernel, d_image, d_output,imageWidth, imageHeight, kernel.getKernelWidth());
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "Convolution time, CUDA version: "<<std::fixed<<std::setprecision(10)<<elapsed_seconds.count()<<std::endl;

    //Copy back the result from the GPU
    cudaMemcpy(output.data(),d_output,output.size()* sizeof(float) ,cudaMemcpyDeviceToHost);

    imageMatrix = output;

    //Free the memory
    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

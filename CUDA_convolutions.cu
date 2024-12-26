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

    int midKernel = kernelSize / 2;
    //Index for calculation
    float sum = 0.0;

    if(x >= midKernel && x < width - midKernel &&
            y >= midKernel && y < height - midKernel) {

        for(int ki = -midKernel; ki <= midKernel; ki++) {
            for(int kj = -midKernel; kj <= midKernel; kj++) {
                float imgValue = mat[(ki + y) * width + (kj + x)];
                float kernelValue = ker[(ki + midKernel) * kernelSize + (kj + midKernel)];
                sum += imgValue * kernelValue;
            }
        }

        //Thresholding pixels
        if(sum < 0){
            sum = 0;
        }
        else if (sum > 255){
            sum = 255;
        }

        output[y * width + x] = sum;
        sum = 0;
    }

}
void applyCUDAConvolution(std::vector<float> &imageMatrix, const kernel &kernel, std::vector<float> &output, int imageWidth, int imageHeight){
    //requiresPadding
    std::chrono::time_point<std::chrono::system_clock> start, end;

    //Set up grid and blocks
    dim3 blockDim(32,32);
    dim3 gridDim((imageWidth + blockDim.x - 1) / blockDim.x, (imageHeight + blockDim.y - 1) / blockDim.y);

    start = std::chrono::system_clock::now();

    //Allocate device memory
    float *d_image, *d_kernel, *d_output;
    cudaMalloc(reinterpret_cast<void**> (&d_image), imageMatrix.size()*sizeof(float));
    cudaMalloc(reinterpret_cast<void**> (&d_kernel), kernel.getKernel().size()*sizeof(float));
    cudaMalloc(reinterpret_cast<void**> (&d_output), output.size()*sizeof(float));

    //Transfer data from host to device memory
    cudaMemcpy(d_image, imageMatrix.data(),imageMatrix.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.getKernel().data(),kernel.getKernel().size() * sizeof(float), cudaMemcpyHostToDevice);
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds1 = end-start;
    std::cout << "H2D time, CUDA version: "<<std::fixed<<std::setprecision(10)<<elapsed_seconds1.count()<<std::endl;

    //Call kernel func
    start = std::chrono::system_clock::now();
    applyCUDAConvolutionKernel<<<gridDim, blockDim>>>(d_kernel, d_image, d_output,imageWidth, imageHeight, kernel.getKernelWidth());

    //Waits for threads to finish
    cudaDeviceSynchronize();
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds2 = end-start;
    std::cout << "Convolution time, CUDA version: "<<std::fixed<<std::setprecision(10)<<elapsed_seconds2.count()<<std::endl;

    start = std::chrono::system_clock::now();
    //Copy back the result from the GPU
    cudaMemcpy(output.data(),d_output,output.size()* sizeof(float) ,cudaMemcpyDeviceToHost);

    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds3 = end-start;
    std::cout << "D2H time, CUDA version: "<<std::fixed<<std::setprecision(10)<<elapsed_seconds3.count()<<std::endl;

    //Free the memory
    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_output);

    std::cout<<"Total CUDA version time"<<std::fixed<<std::setprecision(10)<<elapsed_seconds1.count()+elapsed_seconds2.count()+elapsed_seconds3.count()<<std::endl;
}

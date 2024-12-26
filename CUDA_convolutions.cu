#include "CUDA_convolutions.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>

#define BLOCK_WIDTH 	32
#define BLOCK_HEIGHT	32


__global__ void applyGlobalCUDAConvolutionKernel(const float * ker, const float* mat, float* output, int width, int height, int kernelSize) {

    //Calculate global position for each thread
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;


    //We are already iterating thorugh the image, just with the x,y positions

    int midKernel = kernelSize / 2;
    //Index for calculation
    float sum = 0.0;

    // check out of bounds for thread indx
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
    }

}
void applyGlobalCUDAConvolution(std::vector<float> &imageMatrix, const kernel &kernel, std::vector<float> &output, int imageWidth, int imageHeight){
    std::cout << "Starting global CUDA convolution"<<std::endl;

    std::chrono::time_point<std::chrono::system_clock> start, end;
    //Set up grid and blocks
    dim3 blockDim(BLOCK_WIDTH,BLOCK_HEIGHT);
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
    std::cout << "H2D time, global CUDA version: "<<std::fixed<<std::setprecision(10)<<elapsed_seconds1.count()<<std::endl;

    //Call kernel func
    start = std::chrono::system_clock::now();
    applyGlobalCUDAConvolutionKernel<<<gridDim, blockDim>>>(d_kernel, d_image, d_output, imageWidth, imageHeight,
                                                            kernel.getKernelWidth());

    //Waits for threads to finish
    cudaDeviceSynchronize();
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds2 = end-start;
    std::cout << "Convolution time, global CUDA version: "<<std::fixed<<std::setprecision(10)<<elapsed_seconds2.count()<<std::endl;

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

    std::cout<<"Total global CUDA version time: "<<std::fixed<<std::setprecision(10)<<elapsed_seconds1.count()+elapsed_seconds2.count()+elapsed_seconds3.count()<<"\n"<<std::endl;
}

//Allocate memory for storing constant kernel vector
__device__ __constant__ float d_constantKernel[25*25];

__global__ void applyConstantCUDAConvolutionKernel(const float* mat, float* output, int width, int height, int kernelSize){
    //Global position
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    int midKernel = kernelSize / 2;
    float sum = 0.0;

    // check out of bounds for thread indx
    if( x >= midKernel && x < width - midKernel &&
            y >= midKernel && y < height - midKernel){

        for(int ki = -midKernel; ki <= midKernel; ki++){
            for(int kj = -midKernel; kj <= midKernel; kj++){
                float imgValue = mat[(ki + y) * width + (kj + x)];
                float kernelValue = d_constantKernel[(ki + midKernel) * kernelSize + (kj + midKernel)];
                //Using the kernel stored in constant memory
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
    }
}

void applyConstantCUDAConvolution(std::vector<float> &imageMatrix, const kernel &kernel, std::vector<float> &output,
                                  int imageWidth, int imageHeight) {
    std::cout <<"Starting constant CUDA convolution"<<std::endl;

    std::chrono::time_point<std::chrono::system_clock> start, end;
    //Set up grid and blocks
    dim3 blockDim(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 gridDim((imageWidth + blockDim.x - 1) / blockDim.x, (imageHeight + blockDim.y - 1) / blockDim.y);

    start = std::chrono::system_clock::now();

    //Allocate device memory
    //No need to allocate memory for the kernel
    float *d_image, *d_output;

    cudaMalloc(reinterpret_cast<void**> (&d_image), imageMatrix.size()*sizeof(float));
    cudaMalloc(reinterpret_cast<void**> (&d_output), output.size()*sizeof(float));

    //Transfer data from host to device memory
    cudaMemcpy(d_image, imageMatrix.data(),imageMatrix.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_constantKernel,kernel.getKernel().data(),kernel.getKernel().size()*sizeof(float), 0, cudaMemcpyHostToDevice);
    //TODO:Understand what it does
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds1 = end-start;
    std::cout << "H2D time, constant CUDA version: "<<std::fixed<<std::setprecision(10)<<elapsed_seconds1.count()<<std::endl;

    //Call constant kernel func
    start = std::chrono::system_clock::now();
    applyConstantCUDAConvolutionKernel<<<gridDim, blockDim>>>(d_image, d_output, imageWidth, imageHeight,
                                                              kernel.getKernelHeight());
    //Waits for threads to finish
    cudaDeviceSynchronize();
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds2 = end-start;
    std::cout << "Convolution time, constant CUDA version: "<<std::fixed<<std::setprecision(10)<<elapsed_seconds2.count()<<std::endl;

    //Copy back result from device to host
    start = std::chrono::system_clock::now();
    cudaMemcpy(output.data(),d_output,output.size()*sizeof(float), cudaMemcpyDeviceToHost);
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds3 = end-start;
    std::cout << "D2H time, constant CUDA version: "<<std::fixed<<std::setprecision(10)<<elapsed_seconds3.count()<<std::endl;

    //free device memory
    cudaFree(d_image);
    cudaFree(d_output);

    std::cout<<"Total constant CUDA version time: "<<std::fixed<<std::setprecision(10)<<elapsed_seconds1.count()+elapsed_seconds2.count()+elapsed_seconds3.count()<<"\n"<<std::endl;
}

__global__ void applySharedCUDAConvolutionKernel(const float* mat, float* output, int width, int height, int kernelSize){
    //Global position
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    //thread indx in block
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    int midKernel = kernelSize / 2;

    //Shared memory dim
    extern __shared__ float sharedMemory[];

    //Index in shared block
    unsigned int sharedWidth = blockDim.x + 2 * midKernel;
    unsigned int sharedIndex = (ty + midKernel) * sharedWidth + (tx + midKernel);

    //Load of needed near pixel, with bounds
    if( x < width && y < height){
        sharedMemory[sharedIndex] = mat[y * width + x];
    }else{
        sharedMemory[sharedIndex] = 0.0f;
    }

    //Load of needed extra bounds
    if(tx < midKernel){
        //Left bound
        sharedMemory[sharedIndex - midKernel] = (x >= midKernel) ? mat[y * width + x - midKernel] : 0.0f;
        //Right bound
        sharedMemory[sharedIndex + blockDim.x] = (x + blockDim.x < width) ? mat[y * width + x + blockDim.x] : 0.0f;
    }
    if (ty < midKernel) {
        //Upper bound
        sharedMemory[sharedIndex - midKernel * sharedWidth] = (y >= midKernel) ? mat[(y - midKernel) * width + x] : 0.0f;
        //Lower bound
        sharedMemory[sharedIndex + blockDim.y * sharedWidth] = (y + blockDim.y < height) ? mat[(y + blockDim.y) * width + x] : 0.0f;
    }

    //Synchronize all thread to make sure that the shared memory is loaded
    __syncthreads();

    if (x >= midKernel && x < width - midKernel && y >= midKernel && y < height - midKernel) {
        float sum = 0.0f;

        for (int ki = -midKernel; ki <= midKernel; ki++) {
            for (int kj = -midKernel; kj <= midKernel; kj++) {
                float imgValue = sharedMemory[(ty + midKernel + ki) * sharedWidth + (tx + midKernel + kj)];
                float kernelValue = d_constantKernel[(ki + midKernel) * kernelSize + (kj + midKernel)];
                sum += imgValue * kernelValue;
                //Using both shared and constant memory acces
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
    }
}

void applySharedCUDAConvolution(std::vector<float> &imageMatrix, const kernel &kernel, std::vector<float> &output,
                                int imageWidth, int imageHeight) {
    std::cout <<"Starting shared CUDA convolution"<<std::endl;

    std::chrono::time_point<std::chrono::system_clock> start, end;
    //Set up grid and blocks
    dim3 blockDim(16,16);
    dim3 gridDim((imageWidth + blockDim.x - 1) / blockDim.x, (imageHeight + blockDim.y - 1) / blockDim.y);

    //Compute shared memory needed
    unsigned int sharedMemSize = (blockDim.x + kernel.getKernelWidth() - 1) *
                        (blockDim.y + kernel.getKernelHeight() - 1) * sizeof(float);

    start = std::chrono::system_clock::now();

    //Allocate device memory
    //No need to allocate memory for the kernel
    float *d_image, *d_output;
    cudaMalloc(reinterpret_cast<void**> (&d_image), imageMatrix.size()*sizeof(float));
    cudaMalloc(reinterpret_cast<void**> (&d_output), output.size()*sizeof(float));
    cudaMemcpyToSymbol(d_constantKernel, kernel.getKernel().data(), kernel.getKernel().size()*sizeof(float));

    cudaMemcpy(d_image, imageMatrix.data(), imageMatrix.size()*sizeof(float), cudaMemcpyHostToDevice);
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds1 = end-start;
    std::cout << "H2D time, shared CUDA version: "<<std::fixed<<std::setprecision(10)<<elapsed_seconds1.count()<<std::endl;

    //Call shared kernel func
    start = std::chrono::system_clock::now();
    applySharedCUDAConvolutionKernel<<<gridDim, blockDim, sharedMemSize>>>(d_image, d_output, imageWidth, imageHeight,
                                                                           kernel.getKernelHeight());
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds2 = end-start;
    std::cout << "Convolution time, shared CUDA version: "<<std::fixed<<std::setprecision(10)<<elapsed_seconds2.count()<<std::endl;

    //Copy back result from device to host
    start = std::chrono::system_clock::now();
    cudaMemcpy(output.data(), d_output, output.size()*sizeof(float),cudaMemcpyDeviceToHost);
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds3 = end-start;
    std::cout << "D2H time, shared CUDA version: "<<std::fixed<<std::setprecision(10)<<elapsed_seconds3.count()<<std::endl;

    //Free device memory
    cudaFree(d_image);
    cudaFree(d_output);

    std::cout<<"Total shared CUDA version time: "<<std::fixed<<std::setprecision(10)<<elapsed_seconds1.count()+elapsed_seconds2.count()+elapsed_seconds3.count()<<"\n"<<std::endl;
}


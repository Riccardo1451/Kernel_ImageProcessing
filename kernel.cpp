#include "kernel.h"
#include <iostream>
#include <cmath>

#define MIN_SHARPEN -1
#define MAX_SHARPEN 5
#define MAX_BOOST_SHARPEN 9
#define MIN_EDGE -1
#define MAX_EDGE 8
#define MIN_LAPLACIAN -1
#define MAX_LAPLACIAN 4
#define MAX_LoG 16
#define MED_LoG -2
#define MIN_LoG -1

kernel::kernel() { //default constructor
}

void kernel::printKernel() const {
    //Show kernel values 
    std::cout<< "+---Kernel---+"<<std::endl;

    for (int i = 0; i < kernelHeight; i++){
        for (int j = 0; j < kernelWidth; j++){
            //Acces the linear matrix
            std::cout<<kernelMatrix[i*kernelWidth + j]<<"   ";
        }
        std::cout << std::endl;
    }

    std::cout << "+------------+"<<std::endl;
}

void kernel::kernelBase(std::vector<float>& kernel, int baseWidth, int baseHeight, int min, int max){
    for(int i = 0; i < baseHeight; i++){
        for(int j = 0; j < baseWidth; j++){
            if(i == static_cast<int>(std::ceil(baseHeight / 2)) && 
                            j == static_cast<int>(std::ceil(baseWidth / 2))){
                //middle value
                kernel[i*baseWidth + j] = max;
            }else{
                kernel[i*baseWidth + j] = min;
            }
        }
    }
}

void kernel::setGaussianFilter(const int height, const int width, const int stdDev) {
    std::vector<float> kernel(height*width);
    float sum = 0.0;

    //Store mid values
    int midHeight = height / 2;
    int midWidth = width / 2;

    //Consider [0,0] the center of the matrix, to compute the cel values
    for(int i = -midHeight; i < midHeight; i++){
        for(int j = -midWidth; j < midWidth; j++){
            float cellvalue = exp(-(i*i + j*j) / (2*stdDev*stdDev)) / (2*M_PI*stdDev*stdDev);

            //Find the correct matrix spot
            kernel[(i+midHeight)*width + (j+midWidth)] = cellvalue;

            //sum all cel values to normalize
            sum += cellvalue;
        }
    }
    //Normalize dividing all elements for the total
    for(auto &el : kernel){
        el /= sum;
    }

    //Save state
    kernelMatrix = kernel;
    kernelHeight = height;
    kernelWidth = width;
}

void kernel::setSharpenFilter(bool high_boost) {
    //Allocate vector
    std::vector<float> kernel(3*3);
    if(high_boost){
        //Stronger alternative
        kernelBase(kernel,3,3,MIN_SHARPEN,MAX_BOOST_SHARPEN);
    } else{
        //Classic sharpen
        kernelBase(kernel,3,3,MIN_SHARPEN,MAX_SHARPEN);
        //Set zeros in angles
        kernel[0] = 0;
        kernel[2] = 0;
        kernel[6] = 0;
        kernel[8] = 0;
        }
    //Save state
    kernelMatrix = kernel;
    kernelHeight = 3;
    kernelWidth = 3;
}

void kernel::setEdgeDetectionFilter() {
    //Allocate vector
    std::vector<float> kernel(3*3);

    kernelBase(kernel, 3, 3, MIN_EDGE, MAX_EDGE);

    //Save state
    kernelMatrix = kernel;
    kernelHeight = 3;
    kernelWidth = 3;
}

void kernel::setLaplacianFilter() {
    //Allocate vector
    std::vector<float> kernel(3*3);

    kernelBase(kernel, 3, 3, MIN_LAPLACIAN, MAX_LAPLACIAN);

    //Set zeros in angles
    kernel[0] = 0;
    kernel[2] = 0;
    kernel[6] = 0;
    kernel[8] = 0;

    //Save state
    kernelMatrix = kernel;
    kernelHeight = 3;
    kernelWidth = 3;
}

void kernel::setGaussianLaplacianFilter() {
    //Allocate vector
    std::vector<float> kernel(5 * 5);

    //Manually set values
    kernel[12] = MAX_LoG;
    kernel[2] = MED_LoG;
    kernel[6] = MED_LoG;
    kernel[8] = MED_LoG;
    kernel[10] = MED_LoG;
    kernel[14] = MED_LoG;
    kernel[16] = MED_LoG;
    kernel[18] = MED_LoG;
    kernel[22] = MED_LoG;
    kernel[7] = MIN_LoG;
    kernel[11] = MIN_LoG;
    kernel[13] = MIN_LoG;
    kernel[17] = MIN_LoG;

    //Save state
    kernelMatrix = kernel;
    kernelHeight = 5;
    kernelWidth = 5;
}


//Getter
int kernel::getKernelHeight() const{
    return kernelHeight;
}
int kernel::getKernelWidth() const{
    return kernelWidth;
}
std::vector<float> kernel::getKernel() const{
    return kernelMatrix;
}
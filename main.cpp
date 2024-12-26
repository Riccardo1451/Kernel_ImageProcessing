#include <iostream>
#include "kernel.h"
#include "image.h"


int main(){
    kernel ker = kernel();
    ker.setLaplacianFilter();
    ker.printKernel();

    std::string outpath ="/data01/pc24ricfan/Desktop/Kernel_Code/images/out/"+ker.getKernelInfo()+".png";
    std::string outpathCUDA = "/data01/pc24ricfan/Desktop/Kernel_Code/images/outCUDA/"+ker.getKernelInfo()+"CUDA.png";
    image im = image();

    im.loadImage("/data01/pc24ricfan/Desktop/Kernel_Code/images/in/3.png");
    im.applyConvolution(ker,true); //True for replicate padding, false for zero padding
    im.saveImage(outpath);

    im.loadImage("/data01/pc24ricfan/Desktop/Kernel_Code/images/in/3.png");
    im.applyCUDAConvolution(ker,true);
    im.saveImage(outpathCUDA);

}

#include <iostream>
#include "kernel.h"
#include "image.h"


int main(){
    kernel ker = kernel();
    ker.setEdgeDetectionFilter();
    ker.printKernel();
    std::string outpath ="/data01/pc24ricfan/Desktop/Kernel_Code/images/out/"+ker.getKernelInfo()+".png";

    image im = image();

    im.loadImage("/data01/pc24ricfan/Desktop/Kernel_Code/images/in/3.png");
    im.applyConvolution(ker,false); //True for replicate padding, false for zero padding
    im.saveImage(outpath);

    im.loadImage("/data01/pc24ricfan/Desktop/Kernel_Code/images/in/3.png");
    im.applyCUDAConvolution(ker,false);
    im.saveImage(outpath);
}

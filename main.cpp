#include <iostream>
#include "kernel.h"
#include "image.h"


int main(){
    kernel ker = kernel();
    ker.setEdgeDetectionFilter();
    ker.printKernel();


    image im = image();
    im.loadImage("/Users/riccardofantechi/Desktop/Universita/Quarto anno/Parrallel Programming/Kernel/Kernel_Code/images/in/3.png");
    im.applyConvolution(ker,false); //True for replicate padding, false for zero padding
    std::string outpath ="/Users/riccardofantechi/Desktop/Universita/Quarto anno/Parrallel Programming/Kernel/Kernel_Code/images/out/"+ker.getKernelInfo()+".png";
    im.saveImage(outpath);

}

#include <iostream>
#include "kernel.h"
#include "image.h"
int main(){
    kernel ker = kernel();
    ker.setGaussianLaplacianFilter();
    ker.printKernel();

    image im = image();
    im.loadImage("/Users/riccardofantechi/Desktop/Universita/Quarto anno/Parrallel Programming/Kernel/Kernel_Code/images/in/Lichtenstein_img_processing_test.png");
    im.saveImage("/Users/riccardofantechi/Desktop/Universita/Quarto anno/Parrallel Programming/Kernel/Kernel_Code/images/out/imagetest.png");
}

#include <iostream>
#include "kernel.h"
int main(){
    kernel ker = kernel();
    ker.setGaussianLaplacianFilter();
    ker.printKernel();
}

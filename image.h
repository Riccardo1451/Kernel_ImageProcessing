#ifndef IMAGE_H
#define IMAGE_H
#include <vector>
#include "kernel.h"
#include <iostream>



class image {
public:
    image() = default; //Ctor default
    ~image(); //Dtor

    void loadImage(std::string pathImage);
    void saveImage(std::string pathImage);
    void addPadding(int kernelHeight, bool replicate);
    void applyConvolution(const kernel& kernel, bool padding);

    //CUDA wrap
    void applyGlobalCUDAConvolution(const kernel& kernel, bool padding);
    void applyConstantCUDAConvolution(const kernel& kernel, bool padding);
    void applySharedCUDAConvolution(const kernel& kernel, bool padding);
    //Getter
    std::vector<float> getImageMatrix() const;
    int getImageHeight() const;
    int getImageWidth() const;

private:

    std::vector<float> imageMatrix;
    int imageHeight;
    int imageWidth;

};

#endif // IMAGE_H

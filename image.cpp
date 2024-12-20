#include "image.h"
#include "CUDA_convolutions.cuh"
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <omp.h>

image::~image(){
    imageMatrix.clear();
}

void image::loadImage(const std::string pathImage) {
    //Create image obj from path
    cv::Mat image = cv::imread(pathImage, cv::IMREAD_GRAYSCALE);

    if(image.empty()) {
        std::cout << "Error loading the image from the specified path"<<std::endl;
        return;
    }

    //Allocate tmp vector
    std::vector<float>tmpimage(image.rows*image.cols);

    //Transform image to vector
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            tmpimage[i*image.cols + j] = static_cast<float>(image.at<uchar>(i, j));
        }
    }

    //Save state
    imageMatrix = tmpimage;
    imageHeight = image.rows;
    imageWidth = image.cols;
}
void image::saveImage(const std::string pathImage) {
    //Convert vector to Mat element
    cv::Mat image(imageHeight, imageWidth, CV_32F, imageMatrix.data()); //CV_32F tells that the values are float

    //Normalize values between 0 and 255
    cv::Mat normalizedImage;
    cv::normalize(image, normalizedImage, 0, 255, cv::NORM_MINMAX);

    //Convert to CV_8U int values between 0 and 255
    normalizedImage.convertTo(normalizedImage, CV_8U);

    //Create the image.png in the path
    if(cv::imwrite(pathImage,normalizedImage)){}
    else {
        std::cout << "Error saving the image to: "<< pathImage << std::endl;
        return;
    }
}

void image::addPadding(int const kernelHeight, bool replicate = false) {
    //Compute padding value
    int pad = kernelHeight / 2 ;

    //Set new dimension
    int paddedRows = imageHeight + 2*pad ;
    int paddedCols = imageWidth + 2*pad ;

    //Allocate new image
    std::vector<float> paddedImage (paddedRows*paddedCols , 0.0f); //inizialized with zeros

    //Just copy the original image in the middle of the new one
    for (int i = 0; i < imageHeight; i++) {
        for (int j = 0; j < imageWidth; j++) {
            if( i >= pad && j < imageHeight + pad &&
                j >= pad && j < imageWidth + pad) {
                paddedImage[i * paddedCols + j] = imageMatrix[(i-pad) * imageWidth + (j-pad)];
            } else if(replicate) {
                //Replicate padding using borders
                int src_i = std::max(0, std::min(i - pad, imageHeight -1));
                int src_j = std::max(0, std::min(j - pad, imageWidth -1));
                paddedImage[i*paddedCols + j] = imageMatrix[src_i * imageWidth + src_j];
            }
        }
    }

    //Save the new padded image
    imageMatrix = paddedImage;
    imageHeight = paddedRows;
    imageWidth = paddedCols;
}

void image::applyConvolution(const kernel& kernel, bool padding) {
    //Pad the image based on the kernel to use
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    addPadding(kernel.getKernelHeight(),padding);
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "Execution time to add padding: "<< elapsed_seconds.count() << std::endl;

    //Allocate output vector
    std::vector<float> output (imageHeight*imageWidth , 0.0f);

    //Compute the convolution
    start = std::chrono::system_clock::now();
    //#pragma omp parallel for //TODO: try comment this for sequential version
    for (int i = 0; i < imageHeight; i++) {
        for (int j = 0; j < imageWidth; j++) {
            float sum = 0.0f; //To store the local sum

            //Iterate through the kernel
            for (int ki = 0; ki < kernel.getKernelHeight(); ki++) {
                for (int kj = 0; kj < kernel.getKernelWidth(); kj++) {
                    int row = i + ki;
                    int col = j + kj;
                    sum += imageMatrix[row*imageWidth+col] *
                           kernel.getKernel()[ki*kernel.getKernelWidth()+kj];
                }
            }
            //Save the sum in the output matrix
            output[i*imageWidth + j] = sum;
        }
    }
    end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    std::cout << "Convolution time, sequential version: "<<elapsed_seconds.count()<<std::endl;
    //Save new state
    imageMatrix = output;
}
void image::applyCUDAConvolution(const kernel &kernel, bool padding) {
    addPadding(kernel.getKernelHeight(), padding);
    std::vector<float> tempresult(imageWidth*imageHeight);
    ::applyCUDAConvolution(imageMatrix, kernel, tempresult,imageWidth, imageHeight);
    imageMatrix = tempresult;
}

//GETTER
std::vector<float> image::getImageMatrix() const {
    return imageMatrix;
}

int image::getImageHeight() const {
    return imageHeight;
}

int image::getImageWidth() const {
    return imageWidth;
}
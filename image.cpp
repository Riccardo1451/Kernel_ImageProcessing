#include "image.h"
#include "CUDA_convolutions.cuh"
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <omp.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "writer/stb/stb_image_write.h"

#define STB_IMAGE_IMPLEMENTATION
#include "writer/stb/stb_image.h"

image::~image(){
    imageMatrix.clear();
}

void image::loadImage(const std::string pathImage) {
    /*//Create image obj from path
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

*/
    int width ;
    int height;
    int channels;
    unsigned char* data = stbi_load(pathImage.c_str(), &width, &height, &channels, 1);
    if (!data) {
        std::cerr << "Failed to load image" << std::endl;
        return;
    }
    imageMatrix.resize(width*height);

    for (size_t i = 0; i < width*height; ++i) {
        imageMatrix[i] = static_cast<float>(data[i]) / 255.0f; // Normalizza tra 0 e 1
    }

    //Save state
    imageHeight = height;
    imageWidth = width;


    stbi_image_free(data);

}
void image::saveImage(const std::string pathImage) {
    /*//Convert vector to Mat element
    cv::Mat image(imageHeight, imageWidth, CV_32F, imageMatrix.data()); //CV_32F tells that the values are float

    //Normalize values between 0 and 255
    cv::Mat normalizedImage;
    cv::normalize(image, normalizedImage, 0, 255, cv::NORM_MINMAX);

    //Convert to CV_8U int values between 0 and 255
    normalizedImage.convertTo(normalizedImage, CV_8U);
*/

    std::vector<unsigned char> imageData(imageWidth*imageHeight);

    // Converti i valori da float [0.0, 1.0] a unsigned char [0, 255]
    std::transform(imageMatrix.begin(), imageMatrix.end(), imageData.begin(), [](float value) {
        return static_cast<unsigned char>(std::clamp(value * 255.0f, 0.0f, 255.0f));
    });

    //Create the image.png in the path
    /*if(cv::imwrite(pathImage,normalizedImage, {cv::IMWRITE_JPEG_QUALITY, 95})){}
    else {
        std::cout << "Error saving the image to: "<< pathImage << std::endl;
        return;
    }*/
    stbi_write_png(pathImage.c_str(), imageWidth, imageHeight, 1, imageData.data(), imageWidth);
}

void image::addPadding(int const kernelHeight, bool replicate = false) {
    //Compute padding value
    int pad = floor(kernelHeight / 2 );

    //Set new dimension
    int paddedRows = imageHeight + 2*pad ;
    int paddedCols = imageWidth + 2*pad ;


    //Allocate new image
    std::vector<float> paddedImage (paddedRows*paddedCols , 0.0f); //inizialized with zeros


    //Just copy the original image in the middle of the new one
    for (int i = 0; i <imageHeight ; i++) {
        for (int j = 0; j < imageWidth; j++) {
            int paddedIndex = (i + pad) * paddedCols + (j + pad);
            int originalIndex = i * imageWidth + j;

            paddedImage [paddedIndex] = imageMatrix[originalIndex];
        }
    }

    if(replicate) {
        //Fill the borders replicating near pixels
        for(int i = 0; i < paddedRows; i++){
            for(int j = 0; j < paddedCols; j++){
                if(i < pad){
                    //Upper bound
                    paddedImage[i * paddedCols + j] = paddedImage[pad * paddedCols +j];
                }
                else if( i >= imageHeight + pad){
                    //Lower bound
                    paddedImage[i * paddedCols + j] = paddedImage[(imageHeight + pad - 1) * paddedCols + j];
                }

                if (j < pad){
                    //Left bound
                    paddedImage[i * paddedCols + j] = paddedImage[i * paddedCols + pad];
                }
                else if(j >= imageWidth +pad){
                    //Right bound
                    paddedImage[i * paddedCols + j] = paddedImage[i * paddedCols + (imageWidth + pad - 1)];
                }
            }
        }
    }

    //Save the new padded image
    imageMatrix = paddedImage;
    imageHeight = paddedRows;
    imageWidth = paddedCols;
}

void image::applyConvolution(const kernel& kernel, bool padding) {
    omp_set_num_threads(1);
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
    #pragma omp parallel for //TODO: try comment this for sequential version
    for (int i = 0; i < imageHeight; i++) {
        for (int j = 0; j < imageWidth; j++) {
            float sum = 0.0f; //To store the local sum

            //Iterate through the kernel
            for (int ki = 0; ki < kernel.getKernelHeight(); ki++) {
                for (int kj = 0; kj < kernel.getKernelWidth(); kj++) {
                    int row = i + ki;
                    int col = j + kj;
                    if (row >= 0 && row < imageHeight && col >= 0 && col < imageWidth) {
                        sum += imageMatrix[row * imageWidth + col] * kernel.getKernel()[ki * kernel.getKernelWidth() + kj];
                    }
                }
            }
            //Save the sum in the output matrix
            output[i*imageWidth + j] = sum;
        }
    }
    end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    std::cout << "Convolution time, using "<< omp_get_num_teams()<<" threads is: "<<elapsed_seconds.count()<<"\n"<<std::endl;
    //Save new state
    imageMatrix = output;
}
void image::applyGlobalCUDAConvolution(const kernel &kernel, bool padding) {
    addPadding(kernel.getKernelHeight(), padding);
    std::vector<float> tempresult(imageWidth*imageHeight);
    ::applyGlobalCUDAConvolution(imageMatrix, kernel, tempresult, imageWidth, imageHeight);
    imageMatrix = tempresult;
}
void image::applyConstantCUDAConvolution(const kernel &kernel, bool padding) {
    addPadding(kernel.getKernelHeight(), padding);
    std::vector<float> tempresult(imageWidth*imageHeight);
    ::applyConstantCUDAConvolution(imageMatrix, kernel, tempresult, imageWidth, imageHeight);
    imageMatrix = tempresult;
}
void image::applySharedCUDAConvolution(const kernel &kernel, bool padding) {
    addPadding(kernel.getKernelHeight(), padding);
    std::vector<float> tempresult(imageWidth*imageHeight);
    ::applySharedCUDAConvolution(imageMatrix, kernel, tempresult, imageWidth, imageHeight);
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



#include "image.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

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


std::vector<float> image::getImageMatrix() const {
    return imageMatrix;
}

int image::getImageHeight() const {
    return imageHeight;
}

int image::getImageWidth() const {
    return imageWidth;
}

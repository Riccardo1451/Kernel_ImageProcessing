#ifndef IMAGE_H
#define IMAGE_H
#include <vector>
#include <opencv2/opencv.hpp>

class image {
public:
    image() = default; //Ctor default

    void loadImage(std::string pathImage);
    void saveImage(std::string pathImage);

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

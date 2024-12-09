#include <vector>

class kernel{
    public:
    kernel();

    void printKernel() const; //simple print of the matrix values
    void setGaussianFilter(const int height, const int width, const int stdDev); //set the Gaussian values
    void setSharpenFilter(bool high_boost); //set the Sharpen values
    void setEdgeDetectionFilter(); //set the EdgeDetection values
    void setLaplacianFilter(); //set the Laplacian values
    void setGaussianLaplacianFilter(); //set the Gaussian-Laplacian values

    static void kernelBase(std::vector<float>& kernel, int baseWidth, int baseHeight, int min, int max); //center value with max and the other with min
    
    //Getter
    int getKernelHeight() const;
    int getKernelWidth() const;
    std::vector<float> getKernel() const;
    
    private:
    //Linearized matrix of the kernel values
    std::vector<float> kernelMatrix;
    int kernelHeight;
    int kernelWidth;
};
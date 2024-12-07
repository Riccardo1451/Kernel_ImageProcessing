#include <vector>

class kernel{
    public:
    kernel();

    void printKernel(); //simple print of the matrix values
    void setGaussianFilter(); //set the gaussian values
    void setSharpenFilter();
    void setEdgeDetectionFilter();
    void setLaplacianFilter();
    void setGaussianLaplacianFilter();

    private:
    //Linearized matrix of the kernel values
    std::vector<float> kernelMatrix;
    int kernelHeight;
    int kernelWidth;
};
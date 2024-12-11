# Image Kernel Processing with Performance Analysis

This project focuses on applying image processing techniques using convolutional kernels. The primary goal is to evaluate and compare the performance of sequential, multithreaded, and CUDA-based implementations for image processing tasks.

## Features

- **Image Loading and Saving**: Handles image files in PNG format.
- **Kernel Convolution**: Applies various convolutional kernels to images, such as:
    - Gaussian Blur
    - Laplacian
    - Sharpening
    - Edge Detection
- **Padding Techniques**:
    - Zero Padding
    - Pixel Replication Padding
- **Performance Analysis**:
    - Sequential Processing
    - Multithreaded Processing
    - CUDA-Accelerated Processing (for GPU optimization)

## Prerequisites

- C++17 or later
- OpenCV (for image handling)
- CUDA Toolkit (for GPU acceleration)
- CMake (for build configuration)

## How It Works

1. **Load Image**: Reads the input image into a 2D or flattened vector.
2. **Apply Kernel**: Performs convolution using the specified kernel on the image.
3. **Compare Implementations**: Measures and compares execution times for:
    - Sequential (single-threaded)
    - Multithreaded (CPU parallelism)
    - CUDA (GPU parallelism)
4. **Save Results**: Outputs the processed image and timing data.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. Build the project:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

3. Run the program:
   ```bash
   ./Kernel_Code <input_image> <output_image>
   ```

4. Analyze performance:
   The program outputs timing information for each implementation in the console.

## Example Kernels

### Gaussian Blur (3x3):
```
1  2  1
2  4  2
1  2  1
```

### Edge Detection (3x3):
```
-1 -1 -1
-1  8 -1
-1 -1 -1
```

### Sharpening (3x3):
```
 0 -1  0
-1  5 -1
 0 -1  0
```

## Performance Comparison

- **Sequential**: Baseline implementation using a single thread.
- **Multithreaded**: Optimized using OpenMP for CPU parallelism.
- **CUDA**: GPU-accelerated implementation for maximum performance.

Timing results are outputted to the console and can be used to generate plots for visualization.

## License

This project is licensed under the UNIFI License.

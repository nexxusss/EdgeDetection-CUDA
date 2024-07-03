# EdgeDetection-CUDA

This project implements a basic Sobel edge detection algorithm using CUDA.

### Usage

To compile and run this project, you need to have the following dependencies installed:
- CUDA Toolkit
- OpenCV

Follow these steps to compile and run the program:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/nexxusss/EdgeDetection-CUDA.git
    cd EdgeDetection-CUDA
    ```

2. **Compile the program**:
    ```sh
    nvcc -o edge_detection edge.cu `pkg-config --cflags --libs opencv4`
    ```

3. **Run the program**:
    ```sh
    ./edge_detection path/to/your/image.jpg
    ```

For GPU runtime you can use Google Collab which also supports C/C++ code.

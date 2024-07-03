#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

__global__ void detectEdge(const unsigned char* const grayImage,
                           unsigned char* const edgeImage,
                           int numRows, int numCols)
{
    // takes a gray image and extract the edges

    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (col < numCols && row < numRows){
      int idx = row * numCols + col;

      const int Gx[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
        };
        const int Gy[3][3] = {
            {-1, -2, -1},
            {0, 0, 0},
            {1, 2, 1}
        };

        int Gx_val = 0;
        int Gy_val = 0;

        // convolution
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                int neighborRow = row + i;
                int neighborCol = col + j;

                // Boundary check
                if (neighborRow >= 0 && neighborRow < numRows && neighborCol >= 0 && neighborCol < numCols) {
                    int neighborIdx = neighborRow * numCols + neighborCol;
                    Gx_val += grayImage[neighborIdx] * Gx[i + 1][j + 1];
                    Gy_val += grayImage[neighborIdx] * Gy[i + 1][j + 1];
                }
            }
        }

        int gradientVal = sqrtf((Gx_val * Gx_val) + (Gy_val * Gx_val));

        edgeImage[idx] = static_cast<unsigned char>(gradientVal);
    }


}

int main(int argc, char** argv){

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    int numRows = image.rows;
    int numCols = image.cols;
    size_t numPixels = numRows * numCols;

    unsigned char* d_greyImage;
    unsigned char* d_edgeImage;
    cudaMalloc(&d_greyImage, numPixels * sizeof(unsigned char));
    cudaMalloc(&d_edgeImage, numPixels * sizeof(unsigned char));

    cudaMemcpy(d_greyImage, grayImage.ptr<unsigned char>(), numPixels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    const dim3 blockSize(16, 16, 1);
    const dim3 gridSize((numCols + blockSize.x -1)/blockSize.x,
                        (numRows + blockSize.y -1)/blockSize.y, 1);


    detectEdge<<<gridSize,blockSize>>>(d_greyImage, d_edgeImage, numRows, numCols);

    // retrieve the result from device to host
    unsigned char* h_edgeImage = new unsigned char[numPixels];
    cudaMemcpy(h_edgeImage, d_edgeImage, numPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost); 

    cv::Mat edgeImage(numRows, numCols, CV_8UC1, h_edgeImage);
    cv::imwrite("/content/edge_image.png", edgeImage);

    cudaFree(d_edgeImage);
    cudaFree(d_greyImage);

    printf("The process has successfully ended.");
    return 0;
}
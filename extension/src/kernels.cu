#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "kernels.h"

// CUDA Kernel: Enhances high-luminance pixels (cursor candidates) and suppresses background
__global__ void highlight_cursor_kernel(const unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * channels;

    // Assuming BGR format from OpenCV
    unsigned char b = input[idx];
    unsigned char g = input[idx + 1];
    unsigned char r = input[idx + 2];

    // Calculate luminance
    float luminance = 0.114f * b + 0.587f * g + 0.299f * r;

    // Threshold for "white-ish" things (cursor is usually white/bright)
    if (luminance > 200) {
        // Boost brightness
        output[idx] = (unsigned char)min(255.0f, b * 1.2f);
        output[idx+1] = (unsigned char)min(255.0f, g * 1.2f);
        output[idx+2] = (unsigned char)min(255.0f, r * 1.2f);
    } else {
        // Dim background slightly to increase contrast
        output[idx] = (unsigned char)(b * 0.8f);
        output[idx+1] = (unsigned char)(g * 0.8f);
        output[idx+2] = (unsigned char)(r * 0.8f);
    }
}

void launch_highlight_kernel(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    highlight_cursor_kernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();
}


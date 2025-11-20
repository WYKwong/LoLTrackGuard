#pragma once

// Launches the CUDA kernel to highlight bright cursor-like objects
void launch_highlight_kernel(const unsigned char* d_input, unsigned char* d_output, int width, int height, int channels);


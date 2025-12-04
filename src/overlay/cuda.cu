#include "bitmap.hpp"
#include "overlay.hpp"
#include <iostream>
#include <string>

using std::cin;
using std::cout;
using std::string;

__global__ void kernel(const rgb* img1, const rgb* img2, rgb* result, int height, int width){
    int total_threads = blockDim.x * gridDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int size = width * height;
    while (i < size)
    {
        result[i] = overlay(img1[i], img2[i]);
        i += total_threads;
    }
}

int main()
{
    string filename1 = "img1.bmp";
    string filename2 = "img2.bmp";
    string output = "out.bmp";

    bitmap img1(filename1);
    bitmap img2(filename2);

    int width = img1.get_width();
    int height = img1.get_height();
    int size = width * height;

    rgb* img1_pixels = img1.get_pixels_copy();
    rgb* img2_pixels = img2.get_pixels_copy();

    rgb* result = new rgb[size];

    rgb* d_img1_pixels;
    rgb* d_img2_pixels;
    rgb* d_result;

    int threads_per_block = 256;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;

    cudaMalloc(&d_img1_pixels, size * sizeof(rgb));
    cudaMalloc(&d_img2_pixels, size * sizeof(rgb));
    cudaMalloc(&d_result, size * sizeof(rgb));

    cudaMemcpy(d_img1_pixels, img1_pixels, size * sizeof(rgb), cudaMemcpyHostToDevice);
    cudaMemcpy(d_img2_pixels, img2_pixels, size * sizeof(rgb), cudaMemcpyHostToDevice);

    kernel<<<num_blocks, threads_per_block>>>(d_img1_pixels, d_img2_pixels, d_result, img1.get_height(), img1.get_width());
    cudaDeviceSynchronize();

    cudaMemcpy(result, d_result, size * sizeof(rgb), cudaMemcpyDeviceToHost);

    cudaFree(d_img1_pixels);
    cudaFree(d_img2_pixels);
    cudaFree(d_result);

    bitmap overlayed(width, height, result);
    overlayed.save(output);

    delete[] result;
    return 0;
}
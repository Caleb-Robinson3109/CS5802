#include "bitmap.hpp"
#include "overlay.hpp"
#include <iostream>
#include <string>
#include <vector>

using std::cin;
using std::cout;
using std::string;
using std::vector;

__global__ void kernel(rgb* img1, rgb* img2, rgb* result, int size)
{
    int total_threads = blockDim.x * gridDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

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

    rgb* img1_pixels = img1.get_pixels().data();
    rgb* img2_pixels = img2.get_pixels().data();

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

    kernel<<<num_blocks, threads_per_block>>>(d_img1_pixels, d_img2_pixels, d_result, size);
    cudaDeviceSynchronize();

    cudaMemcpy(result, d_result, size * sizeof(rgb), cudaMemcpyDeviceToHost);

    cudaFree(d_img1_pixels);
    cudaFree(d_img2_pixels);
    cudaFree(d_result);

    bitmap overlayed(width, height, vector<rgb>(result, result + (width * height)));
    overlayed.save(output);

    delete[] result;
    return 0;
}
#include "bitmap.hpp"
#include "overlay.hpp"
#include <iostream>
#include <string>

using std::cin, std::cout, std::string

__global__ void kernel(rgb *img1, rgb* img2, rgb *result, int size){
    int total_threads = blockDim.x * gridDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while(i < size){
        result[i] = overlay(img1[i], img2[i]);
        i += total_threads;
    }
}

int main(){
    string filename1, filename2, output;
    /*
    cout << "Image 1: ";
    cin >> filename1;
    cout << "Image 2: ";
    cin >> filenmae2;
    cout << "Output: ";
    cin >> output;
    */

    filename1 = "img1.bmp";
    filename2 = "img2.bmp";
    output = "out.bmp";
    bitmap img1(filename1);
    bitmap img2(filename2);
    int size = img1.get_width() * img1.get_height();
    rgb* result[size];
    rgb* img1_pixels = img1.get_pixels();
    rgb* img2_pixels = img2.get_height();
    rgb* d_result;
    rgb* d_img1_pixels;
    rgb* d_img2_pixels;
    int num_blocks = 4;
    int threads_per_block = 64;
    
    cudaMalloc((void**) &d_result, sizeof(result));
    cudaMalloc((void**), &d_img1_pixels, sizeof(img1_pixels));
    cudaMalloc((void**), &d_img2_pixels, sizeof(img2_pixels));

    cudaMemcpy(d_img1_pixels, img1_pixels, sizeof(img1_pixels), cudaMemcpyHostToDevice);
    cudaMemcpy(d_img2_pixels, img2_pixels, sizeof(img2_pixels), cudaMemcpyHostToDevice);

    kernel<<<num_blocks, threads_per_block>>>(d_img1_pixels, d_img2_pixels, d_result, size);
    cudaDeviceSynchronize();

    cudaMemcpy(result, d_result, sizeof(result), cudaMemcpyDeviceToHost);

    cudaFree(d_img1_pixels);
    cudaFree(d_img2_pixels);
    cudaFree(d_result);
    
    bitmap overlayed(img1.get_width(), img1.get_height(), result);
    overlayed.save(output);

    return 0;    
}
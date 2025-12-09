#include "bitmap.hpp"
#include "gaussian.hpp"
#include "pathutil.hpp"
#include <iostream>
#include <string>
#include <vector>

using std::cin;
using std::cout;
using std::string;
using std::vector;

__global__ void kernel(double * gaussian_filter, int filter_size, rgb* img_pixels, rgb* result, int height, int width){
    int total_threads = blockDim.x * gridDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int size = width * height;
    // apply filter
    while(i < size){
        int y = i / width;
        int x = i % width;
        result[i] = apply_gaussian_filter(gaussian_filter, filter_size, img_pixels, height, width, x, y);
        i += total_threads;
    }
}

int main()
{

    Timer timer;
    timer.start();
    vector<PathSet> paths = generatePaths();

    for(PathSet currentPaths : paths){
        bitmap img(currentPaths.input1Path);

        int width = img.get_width();
        int height = img.get_height();
        int size = width * height;

        rgb* img_pixels = img.get_pixels_copy();

        rgb* result = new rgb[size];

        rgb* d_img_pixels;
        rgb* d_result;

        int threads_per_block = 256;
        int num_blocks = (size + threads_per_block - 1) / threads_per_block;

        int filter_size = 11;
        int filter_output_size = 0;
        double * gaussian_filter = create_gaussian_filter(filter_output_size, filter_size);
        double * d_gaussian_filter;

        cudaMalloc(&d_img_pixels, size * sizeof(rgb));
        cudaMalloc(&d_result, size * sizeof(rgb));
        cudaMalloc(&d_gaussian_filter, filter_output_size * sizeof(double));

        cudaMemcpy(d_img_pixels, img_pixels, size * sizeof(rgb), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gaussian_filter, gaussian_filter, filter_output_size * sizeof(double), cudaMemcpyHostToDevice);

        kernel<<<num_blocks, threads_per_block>>>(d_gaussian_filter, filter_size, d_img_pixels, d_result, height, width);
        cudaDeviceSynchronize();

        cudaMemcpy(result, d_result, size * sizeof(rgb), cudaMemcpyDeviceToHost);

        cudaFree(d_img_pixels);
        cudaFree(d_result);
        cudaFree(d_gaussian_filter);

        bitmap blurred(width, height, result);
        blurred.save(currentPaths.outputPath);

        delete[] result;
    }

    timer.stop();
    std::cout << "Cuda Gaussian Blur Runtime (ms): " << timer.get_duration_ms() << "\n";

    return 0;
}

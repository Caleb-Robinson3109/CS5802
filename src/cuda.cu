#include "bitmap.hpp"
#include "overlay.hpp"

__global__ void kernel(rgb *img1, rgb* img2, rgb *result, int size){
    int total_threads = blockDim.x * gridDim.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while(i < size){
        result[i] = overlay(img1[i], img2[i]);
        i += total_threads;
    }
}
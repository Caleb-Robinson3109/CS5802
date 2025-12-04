#pragma once

#include "bitmap.hpp"
#include <cstdint>
#include <cmath>

#ifdef __CUDACC__
    #define DEVICE __device__
#else
    #define DEVICE
#endif

//probs cant do this on gpu bc/ of the math? idk tho
int* create_gaussian_filter(const int size, const double sigma = 1);
DEVICE rgb apply_gaussian_filter(const int** filter, const int size, const rgb** image, const int height, const int width, const  int x, const int y);

//filter = gaussian filter
//size = size of the filter -> size x size
//image = pointer to the pixels
//height = height of the image
//width = width of the image
//x = x pos of pixel to apply filter
//y = y pos of pixel to apply filter 
DEVICE rgb apply_gaussian_filter(const int** filter, const int size, const rgb** image, const int height, const int width, const  int x, const int y){
    
}

//filter (x,y) = filter[y * size + x]
int* create_gaussian_filter(const int size, const double sigma = 1){
    double center = (size - 1) / 2.0;
    double sum = 0.0;
    int* filter = new int[size * size];

    for(int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            double x = i - center;
            double y = j - center;
            filter[j * size + i] = std::exp(-(x*x + y*y) / (2 * sigma * sigma));
            sum += filter[j * size + i];
        }
    }

    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            filter[j * size + i] /= sum;
        }
    }
    return filter;
}
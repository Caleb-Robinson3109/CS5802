#pragma once

#include "bitmap.hpp"
#include <cstdint>

#ifdef __CUDACC__
    #define DEVICE __device__
#else
    #include <math.h>
    #define DEVICE
#endif

//probs cant do this on gpu bc/ of the math? idk tho
double* create_gaussian_filter(int size, const double sigma);
DEVICE rgb apply_gaussian_filter(const double* filter, int size, const rgb* image, const int height, const int width, const  int x, const int y);

//filter = gaussian filter
//size = size of the filter -> size x size
//image = pointer to the pixels
//height = height of the image
//width = width of the image
//x = x pos of pixel to apply filter
//y = y pos of pixel to apply filter 
//x ~ width,  y ~ height
DEVICE rgb apply_gaussian_filter(const double* filter, int size, const rgb* image, const int height, const int width, const  int x, const int y){
    rgb result;
    double result_r = 0.0;
    double result_g = 0.0;
    double result_b = 0.0;
    size = size % 2 == 0 ? size + 1 : size;
    int half_size = size / 2;
    for(int filter_y = 0; filter_y < size; filter_y++){
        for(int filter_x = 0; filter_x < size; filter_x++){
            //check to make sure not out of bounds of image
            int image_x = (x - half_size + filter_x) < 0 ? 0 : (x - half_size + filter_x) >= width ? width - 1 : (x - half_size + filter_x);
            int image_y = (y - half_size + filter_y) < 0 ? 0 : (y - half_size + filter_y) >= height ? height - 1 : (y - half_size + filter_y);

            result_r += image[image_y * width + image_x].r * filter[filter_y * size + filter_x];
            result_g += image[image_y * width + image_x].g * filter[filter_y * size + filter_x];
            result_b += image[image_y * width + image_x].b * filter[filter_y * size + filter_x];
        }
    }

    // switched to c round function because it will be converted to a device intrinsic when compiled with cuda.
    result.r = (uint8_t)round(result_r);
    result.g = (uint8_t)round(result_g);
    result.b = (uint8_t)round(result_b);

    return result;
}

//filter (x,y) = filter[y * size + x]
double* create_gaussian_filter(int size, const double sigma = 3.0){
    size = size % 2 == 0 ? size + 1 : size;
    double center = (size - 1) / 2.0;
    double sum = 0.0;
    double* filter = new double[size * size];

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

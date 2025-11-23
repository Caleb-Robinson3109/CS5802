#pragma once

#include "bitmap.hpp"
#include <cstdint>

__device__ rgb overlay(rgb base, rgb layer);
__device__ uint8_t overlay(uint8_t base, uint8_t layer);

__device__ rgb overlay(rgb base, rgb layer){
    rgb result;
    result.r = overlay(base.r, layer.r);
    result.g = overlay(base.g, layer.g);
    result.b = overlay(base.b, layer.b);
    return result;
}

__device__ uint8_t overlay(uint8_t base, uint8_t layer){
    if(base < 128){
        return (2 * base * layer) / 255;
    }
    else{
        return 255 - ((2*(255 - base) * (255 - layer)) / 255);
    }
}
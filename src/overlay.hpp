#pragma once

#ifdef __CUDACC__
    #define DEVICE __device__
#else
    #define DEVICE
#endif

#include "bitmap.hpp"
#include <cstdint>

DEVICE rgb overlay(rgb base, rgb layer);
DEVICE uint8_t overlay(uint8_t base, uint8_t layer);

DEVICE rgb overlay(rgb base, rgb layer){
    rgb result;
    result.r = overlay(base.r, layer.r);
    result.g = overlay(base.g, layer.g);
    result.b = overlay(base.b, layer.b);
    return result;
}

DEVICE uint8_t overlay(uint8_t base, uint8_t layer){
    return (base < 128)
        ? (uint8_t)((2 * base * layer) / 255)
        : (uint8_t)(255 - ((2*(255 - base) * (255 - layer)) / 255));
}
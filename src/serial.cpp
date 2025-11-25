#include "bitmap.hpp"
#include "overlay.hpp"
#include <iostream>
#include <string>

int main(){

    // take in bitmaps
    std::string filename1 = "img1.bmp";
    std::string filename2 = "img2.bmp";
    std::string output = "out.bmp";

    bitmap img1(filename1);
    bitmap img2(filename2);

    int width = img1.get_width();
    int height = img1.get_height();
    int size = width * height;

    rgb* img1_pixels = img1.get_pixels_copy();
    rgb* img2_pixels = img2.get_pixels_copy();

    rgb* result = new rgb[size];

    // apply filter
    for (int i = 0; i < size; i++) {
        result[i] = overlay(img1_pixels[i], img2_pixels[i]);
    }

    // output final bitmap
    bitmap overlayed(width, height, result);
    overlayed.save(output);

    delete[] result;

    return 0;
}

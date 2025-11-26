#include "bitmap.hpp"
#include "overlay.hpp"
#include <iostream>
#include <string>

#define NUM_IMAGES 100

int main(){

    for(int i = 1; i <= NUM_IMAGES; i++){
        // take in bitmaps
        std::string filename1 = "img1/img1_" + std::to_string(i) + ".bmp";
        std::string filename2 = "img2/img2_" + std::to_string(i) + ".bmp";
        std::string output = "out_img/out_" + std::to_string(i) + ".bmp";

        bitmap img1(filename1);
        bitmap img2(filename2);

        int width = img1.get_width();
        int height = img1.get_height();
        int size = width * height;

        rgb* img1_pixels = img1.get_pixels_copy();
        rgb* img2_pixels = img2.get_pixels_copy();

        rgb* result = new rgb[size];

        // apply filter
        for (int j = 0; j < size; j++) {
            result[j] = overlay(img1_pixels[j], img2_pixels[j]);
        }

        // output final bitmap
        bitmap overlayed(width, height, result);
        overlayed.save(output);

        delete[] result;
    }
    return 0;
}

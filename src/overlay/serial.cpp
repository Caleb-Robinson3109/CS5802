#include "bitmap.hpp"
#include "overlay.hpp"
#include "pathutil.hpp"
#include <iostream>
#include <string>
#include <vector>

int main(){

    std::vector<PathSet> paths = generatePaths();

    for(PathSet currentPaths : paths) {

        bitmap img1(currentPaths.input1Path);
        bitmap img2(currentPaths.input2Path);

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
        overlayed.save(currentPaths.outputPath);

        delete[] result;
    }
    return 0;
}

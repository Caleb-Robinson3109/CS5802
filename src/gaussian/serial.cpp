#include "bitmap.hpp"
#include "gaussian.hpp"
#include "pathutil.hpp"
#include <iostream>
#include <string>
#include <vector>

int main(){

    std::vector<PathSet> paths = generatePaths();

    for(PathSet currentPaths : paths) {

        bitmap img(currentPaths.input1Path);

        int width = img.get_width();
        int height = img.get_height();
        int size = width * height;

        rgb* img_pixels = img.get_pixels_copy();

        rgb* result = new rgb[size];

        double* gaussian_filter = create_gaussian_filter(size, 11);

        // apply filter
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                apply_gaussian_filter(gaussian_filter, size, img_pixels, height, width, x, y);
            }
        }

        // output final bitmap
        bitmap overlayed(width, height, result);
        overlayed.save(currentPaths.outputPath);

        delete[] result;
    }
    return 0;
}

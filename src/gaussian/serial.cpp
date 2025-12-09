#include "bitmap.hpp"
#include "gaussian.hpp"
#include "pathutil.hpp"
#include <iostream>
#include <string>
#include <vector>

int main(){

    Timer timer;
    timer.start();
    std::vector<PathSet> paths = generatePaths();

    for(PathSet currentPaths : paths) {

        bitmap img(currentPaths.input1Path);

        int width = img.get_width();
        int height = img.get_height();
        rgb* img_pixels = img.get_pixels_copy();
        rgb* result = new rgb[width * height];

        int filter_size = 11;

        int filter_output_size = 0;
        double* gaussian_filter = create_gaussian_filter(filter_output_size, filter_size);

        // apply filter
        int i = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                result[i] = apply_gaussian_filter(gaussian_filter, filter_size, img_pixels, height, width, x, y);
                i++;
            }
        }

        // output final bitmap
        bitmap blurred(width, height, result);
        blurred.save(currentPaths.outputPath);

        delete[] result;
    }
    timer.stop();
    std::cout << "Serial Gaussian Blur Runtime (ms): " << timer.get_duration_ms() << "\n";
    return 0;
}

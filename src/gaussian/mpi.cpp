#include "bitmap.hpp"
#include "gaussian.hpp"
#include "pathutil.hpp"
#include "timerutil.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>

#include <mpi.h>

int main(int argc, char* argv[]){
    Timer timer;
    timer.start();

    MPI_Init (&argc, &argv);
    int nprocs;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    std::vector<PathSet> paths = generatePaths();

    // Cyclic partitioning
    for (int i = rank; i < paths.size(); i += nprocs) {

        PathSet currentPaths = paths[i];
        std::cout << "rank " << rank << ": \n";
        std::cout << "input 1 :" << currentPaths.input1Path << "\n";
        std::cout << "input 2 :" << currentPaths.input2Path << "\n";
        std::cout << "output :" << currentPaths.outputPath << "\n";
        std::cout << std::endl;

        bitmap img(currentPaths.input1Path);

        int width = img.get_width();
        int height = img.get_height();
        rgb* img_pixels = img.get_pixels_copy();
        rgb* result = new rgb[width * height];

        int filter_size = 11;

        int filter_output_size = 0;
        double* gaussian_filter = create_gaussian_filter(filter_output_size, filter_size);

        // apply filter
        int j = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                result[j] = apply_gaussian_filter(gaussian_filter, filter_size, img_pixels, height, width, x, y);
                j++;
            }
        }

        // output final bitmap
        bitmap blurred(width, height, result);
        blurred.save(currentPaths.outputPath);

        delete[] result;
        delete[] gaussian_filter;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    
    timer.stop();

    if(rank == 0)
        std::cout << "MPI Gaussian Blur Runtime (ms): " << timer.get_duration_ms() << "\n";
    
    return 0;
}

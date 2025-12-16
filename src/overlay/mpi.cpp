#include "bitmap.hpp"
#include "overlay.hpp"
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

    timer.stop();
    std::cout << "MPI Overlay Runtime (ms): " << timer.get_duration_ms() << "\n";

    MPI_Finalize();
    return 0;
}

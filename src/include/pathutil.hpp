#include <string>
#include <vector>

#pragma once

#define NUM_IMAGES 100

struct PathSet {
    std::string input1Path;
    std::string input2Path;
    std::string outputPath;
};

std::vector<PathSet> generatePaths(){
    std::vector<PathSet> returnedPaths;
    for(int i = 1; i <= NUM_IMAGES; i++){
        PathSet newPath = {
            .input1Path = "img1/img1_" + std::to_string(i) + ".bmp",
            .input2Path = "img2/img2_" + std::to_string(i) + ".bmp",
            .outputPath = "out_img/out_" + std::to_string(i) + ".bmp",
        };
        returnedPaths.push_back(newPath);
    }
    return returnedPaths;
}

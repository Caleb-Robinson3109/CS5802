#pragma once

#include <cstdint>
#include <string>
#include <fstream>
#include <iostream>

#pragma pack(push,1)

struct bmp_file_header{
    uint16_t bfh_type;
    uint32_t bfh_size;
    uint16_t bfh_reserved_1;
    uint16_t bfh_reserved_2;
    uint32_t bfh_off_bits;
};

struct bmp_info_header{
    uint32_t bih_size;
    int32_t bih_width;
    int32_t bih_height;
    uint16_t bih_planes;
    uint16_t bih_bit_count;
    uint32_t bih_compression;
    uint32_t bih_size_image;
    int32_t bih_x_pels_per_meter;
    int32_t bih_y_pel_per_meter;
    uint32_t bih_clr_used;
    uint32_t bih_clr_imported;
};

struct rgb{
    uint8_t b;
    uint8_t g;
    uint8_t r;
};

#pragma pack(pop)

class bitmap{
private:
    int height;
    int width;
    rgb* pixels;

public:
    bitmap(const std::string filename);
    bitmap(const int height, const int width, rgb* pixels);

    ~bitmap();

    int get_width() const;
    int get_height() const;
    rgb* get_pixels() const;
    rgb* get_pixels_copy();

    void set_image(const int height, const int width, const rgb* pixels);

    void load(const std::string filename);    
    void save(const std::string filename);
};

bitmap::bitmap(const std::string filename){
    this->load(filename);
}

bitmap::bitmap(const int height, const int width, rgb* pixels){
    this->width = width;
    this->height = height;
    this->pixels = new rgb[height * width];
    for(int h = 0; h < height; h++){
        for(int w = 0; w < width; w++){
            this->pixels[h * width + w] = pixels[h * width + w];
        }
    }
}

bitmap::~bitmap(){
    delete[] pixels;
}

int bitmap::get_width() const { return width; }

int bitmap::get_height() const { return height; }

rgb* bitmap::get_pixels() const { return pixels; }

rgb* bitmap::get_pixels_copy(){
    rgb* copy = new rgb[width * height];
    for(int i = 0; i < height * width; i++){
        copy[i] = pixels[i];
    }
    return copy;
}

void bitmap::set_image(const int height, const int width, const rgb* pixels){
    if((this->height != height) || (this->width != width)){
        std::cerr << "Must have the same height x width of orginal image!" << std::endl;
        return;
    }
    for(int h = 0; h < height; h++){
        for(int w = 0; w < width; w++){
            this->pixels[h * width + w] = pixels[h * width + w];
        }
    }
}

void bitmap::load(const std::string filename){
    std::ifstream file(filename, std::ios::binary);
    if(!file){
        std::cerr << "Failed to open file!" <<  std::endl;
        return;
    }

    bmp_file_header file_header;
    bmp_info_header info_header;

    file.read(reinterpret_cast<char *>(&file_header), sizeof(file_header));
    file.read(reinterpret_cast<char *>(&info_header), sizeof(info_header));

    if(file_header.bfh_type != 0x4d42){
        std::cerr << "Not a .bmp file!" << std::endl;
        return;
    }

    if(info_header.bih_bit_count != 24){
        std::cerr << "Only 24 bit .bmp please!" << std::endl;
        return;
    }

    width = info_header.bih_width;
    height = info_header.bih_height;

    int row_padding = (4 - (width * 3) % 4) % 4;

    pixels = new rgb[height * width];

    file.seekg(file_header.bfh_off_bits, std::ios::beg);

    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
            rgb pixel;
            file.read(reinterpret_cast<char*>(&pixel), sizeof(rgb));
            pixels[y * width + x] = pixel;
        }
        file.ignore(row_padding);
    }
}

void bitmap::save(const std::string filename){
    std::ofstream file(filename, std::ios::binary);
    if(!file){
        std::cerr << "Failed to make .bmp file!" << std::endl;
        return;
    }

    int row_padding = (4 - (width * 3) % 4) % 4;
    int data_size = (width * 3 + row_padding) * height;

    bmp_file_header file_header{};
    bmp_info_header info_header{};

    file_header.bfh_type = 0x4d42;
    file_header.bfh_size = sizeof(bmp_file_header) + sizeof(bmp_info_header) + data_size;
    file_header.bfh_off_bits = sizeof(bmp_file_header) + sizeof(bmp_info_header);

    info_header.bih_size = sizeof(bmp_info_header);
    info_header.bih_width = width;
    info_header.bih_height = height;
    info_header.bih_planes = 1;
    info_header.bih_bit_count = 24;
    info_header.bih_compression = 0;
    info_header.bih_size_image = data_size;

    file.write(reinterpret_cast<const char*>(&file_header), sizeof(file_header));
    file.write(reinterpret_cast<const char*>(&info_header), sizeof(info_header));

    uint8_t padding[3] = {0, 0, 0};

    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
            rgb pixel = pixels[y * width + x];
            file.write(reinterpret_cast<const char*>(&pixel), sizeof(pixel));
        }
        file.write(reinterpret_cast<const char*>(padding), row_padding);
    }
}
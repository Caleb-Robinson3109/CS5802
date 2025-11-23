#pragma once

#include <cstdint>
#include <vector>
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
    int width = 0;
    int height = 0;
    std::vector<rgb> pixels;
    std::vector<uint8_t> r;
    std::vector<uint8_t> g;
    std::vector<uint8_t> b;

public:
    bitmap(const std::string filename);
    bitmap(const int width, const int height, const std::vector<rgb> pixels);
    bitmap(const int width, const int height, const std::vector<uint8_t> r, const std::vector<uint8_t> g, const std::vector<uint8_t> b);

    int get_width() const;
    int get_height() const;
    std::vector<rgb> get_pixels() const;
    std::vector<uint8_t> get_r() const;
    std::vector<uint8_t> get_g() const;
    std::vector<uint8_t> get_b() const;

    void set_pixels(const std::vector<rgb> pixels);
    void set_r(const std::vector<uint8_t> r);
    void set_g(const std::vector<uint8_t> g);
    void set_b(const std::vector<uint8_t> b);

    void load(const std::string filename);    
    void save(const std::string filename);
};

bitmap::bitmap(const std::string filename){
    this->load(filename);
}

bitmap::bitmap(const int width, const int height, const std::vector<rgb> pixels){
    this->width = width;
    this->height = height;
    this->pixels = pixels;
    r.resize(pixels.size());
    g.resize(pixels.size());
    b.resize(pixels.size());
    for(int i = 0; i < pixels.size(); i++){
        r.at(i) = pixels.at(i).r;
        g.at(i) = pixels.at(i).g;
        b.at(i) = pixels.at(i).b;
    }
}

bitmap::bitmap(const int width, const int height, const std::vector<uint8_t> r, const std::vector<uint8_t> g, const std::vector<uint8_t> b){
    this->width = width;
    this->height = height;
    this->r = r;
    this->g = g;
    this->b = b;
    pixels.resize(r.size());
    for(int i = 0; i < r.size(); i++){
        pixels.at(i).r = r.at(i);
        pixels.at(i).g = g.at(i);
        pixels.at(i).b = b.at(i);
    }
}

int bitmap::get_width() const { return width; }
int bitmap::get_height() const { return height; }
std::vector<rgb> bitmap::get_pixels() const { return pixels; }
std::vector<uint8_t> bitmap::get_r() const { return r; }
std::vector<uint8_t> bitmap::get_g() const { return g; }
std::vector<uint8_t> bitmap::get_b() const { return b; }

void bitmap::set_pixels(const std::vector<rgb> pixels){
    this->pixels = pixels;
    r.resize(pixels.size());
    g.resize(pixels.size());
    b.resize(pixels.size());
    for(int i = 0; i < pixels.size(); i++){
        r.at(i) = pixels.at(i).r;
        g.at(i) = pixels.at(i).g;
        b.at(i) = pixels.at(i).b;
    }
}

void bitmap::set_r(const std::vector<uint8_t> r){
    this->r = r;
    for(int i = 0; i < r.size(); i++){
        pixels.at(i).r = r.at(i);
    }
}

void bitmap::set_g(const std::vector<uint8_t> g){
    this->g = g;
    for(int i = 0; i < g.size(); i++){
        pixels.at(i).g = g.at(i);
    }
}

void bitmap::set_b(const std::vector<uint8_t> b){
    this->b = b;
    for(int i = 0; i < b.size(); i++){
        pixels.at(i).b = b.at(i);
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

    pixels.resize(height * width);
    r.resize(height * width);
    g.resize(height * width);
    b.resize(height * width);

    file.seekg(file_header.bfh_off_bits, std::ios::beg);

    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
            rgb pixel;
            file.read(reinterpret_cast<char*>(&pixel), sizeof(rgb));
            pixels.at(y* width + x) = pixel;
            r.at(y* width + x) = pixel.r;
            g.at(y* width + x) = pixel.g;
            b.at(y* width + x) = pixel.b;
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
            rgb pixel = pixels.at(y * width+ x);
            file.write(reinterpret_cast<const char*>(&pixel), sizeof(pixel));
        }
        file.write(reinterpret_cast<const char*>(padding), row_padding);
    }
}
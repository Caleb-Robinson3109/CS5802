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

#pragma pack(pop)

struct rgb{
    uint8_t b;
    uint8_t g;
    uint8_t r;
};

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

    int get_width() const { return width; }
    int get_height() const { return height; }
    std::vector<rgb> get_pixels() const { return pixels; }
    std::vector<uint8_t> get_r() const { return r; }
    std::vector<uint8_t> get_g() const { return g; }
    std::vector<uint8_t> get_b() const { return b; }

    void set_pixels(const std::vector<rgb> pixels) { this->pixels = pixels; }
    void set_r(const std::vector<uint8_t> r) { this->r = r; }
    void set_g(const std::vector<uint8_t> g) { this->g = g; }
    void set_b(const std::vector<uint8_t> b) { this->b = b; }

    void save(const std::string filename);
};
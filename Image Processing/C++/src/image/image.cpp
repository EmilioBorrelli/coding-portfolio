#include "image.hpp"
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <algorithm>
// stb
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// =======================
// Constructors
// =======================

template<typename T>
Image<T>::Image() {}

template<typename T>
Image<T>::Image(int width, int height, int channels)
    : m_width(width), m_height(height), m_channels(channels),
      m_data(width * height * channels) {}


// =======================
// Accessors
// =======================

template<typename T>
T& Image<T>::at(int x, int y, int c) {
    return m_data[(y * m_width + x) * m_channels + c];
}

template<typename T>
const T& Image<T>::at(int x, int y, int c) const {
    return m_data[(y * m_width + x) * m_channels + c];
}

template<typename T>
int Image<T>::width() const { return m_width; }

template<typename T>
int Image<T>::height() const { return m_height; }

template<typename T>
int Image<T>::channels() const { return m_channels; }

template<typename T>
std::vector<T>& Image<T>::data() { return m_data; }

template<typename T>
const std::vector<T>& Image<T>::data() const { return m_data; }

// =======================
// Loading
// =======================

template<typename T>
bool Image<T>::load(const std::string& path) {
    int w, h, c;

    if constexpr (std::is_same_v<T, uint8_t>) {
        unsigned char* img = stbi_load(path.c_str(), &w, &h, &c, 0);
        if (!img) return false;

        m_width = w;
        m_height = h;
        m_channels = c;
        m_data.assign(img, img + w * h * c);
        stbi_image_free(img);
        return true;
    }

    if constexpr (std::is_same_v<T, uint16_t>) {
        uint16_t* img = stbi_load_16(path.c_str(), &w, &h, &c, 0);
        if (!img) return false;

        m_width = w;
        m_height = h;
        m_channels = c;
        m_data.assign(img, img + w * h * c);
        stbi_image_free(img);
        return true;
    }

    if constexpr (std::is_same_v<T, float>) {
        float* img = stbi_loadf(path.c_str(), &w, &h, &c, 0);
        if (!img) return false;

        m_width = w;
        m_height = h;
        m_channels = c;
        m_data.assign(img, img + w * h * c);
        stbi_image_free(img);
        return true;
    }

    return false;
}


// =======================
// Saving
// =======================

template<typename T>
bool Image<T>::save(const std::string& path) const {

    if constexpr (std::is_same_v<T, uint8_t>) {
        return stbi_write_png(
            path.c_str(),
            m_width,
            m_height,
            m_channels,
            m_data.data(),
            m_width * m_channels
        );
    }

    if constexpr (std::is_same_v<T, uint16_t>) {
        return stbi_write_png(
            path.c_str(),
            m_width,
            m_height,
            m_channels,
            m_data.data(),
            m_width * m_channels * sizeof(uint16_t)
        );
    }

    if constexpr (std::is_same_v<T, float>) {
        // Convert float [0,1] → uint16 for output
        std::vector<uint16_t> tmp(m_data.size());

        for (size_t i = 0; i < m_data.size(); ++i) {
            float v = std::clamp(m_data[i], 0.0f, 1.0f);
            tmp[i] = static_cast<uint16_t>(v * 65535.0f);
        }

        return stbi_write_png(
            path.c_str(),
            m_width,
            m_height,
            m_channels,
            tmp.data(),
            m_width * m_channels * sizeof(uint16_t)
        );
    }

    return false;
}


// =======================
// Explicit instantiations
// =======================

template class Image<uint8_t>;
template class Image<uint16_t>;
template class Image<float>;


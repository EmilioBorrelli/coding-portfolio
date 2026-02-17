// image.hpp
#pragma once
#include <vector>
#include <string>
#include "pixel_traits.hpp"

template<typename T>
class Image {
public:
    Image();
    Image(int width, int height, int channels);

    bool load(const std::string& path);
    bool save(const std::string& path) const;

    inline T& at(int x, int y, int c);
    inline const T& at(int x, int y, int c) const;

    int width() const;
    int height() const;
    int channels() const;

    std::vector<T>& data();
    const std::vector<T>& data() const;
    Image<float> toFloat() const;
    Image<uint16_t> toUint16() const;
private:
    int m_width = 0;
    int m_height = 0;
    int m_channels = 0;
    std::vector<T> m_data;
};

template<typename T>
Image<float> Image<T>::toFloat() const {
    Image<float> out(m_width, m_height, m_channels);
    const float maxVal = PixelTraits<T>::max();

    for (size_t i = 0; i < m_data.size(); ++i) {
        out.data()[i] = static_cast<float>(m_data[i]) / maxVal;
    }
    return out;
};

template<typename T>
Image<uint16_t> Image<T>::toUint16() const
{
    Image<uint16_t> out(m_width, m_height, m_channels);

    const float maxIn  = PixelTraits<T>::max();
    const float maxOut = PixelTraits<uint16_t>::max();

    for (size_t i = 0; i < m_data.size(); ++i) {
        float v = static_cast<float>(m_data[i]) / maxIn;
        v = std::clamp(v, 0.0f, 1.0f);
        out.data()[i] = static_cast<uint16_t>(v * maxOut);
    }

    return out;
};
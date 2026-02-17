#pragma once
#include "../../image/image.hpp"
#include "../../image/pixel_traits.hpp"
#include <stdexcept>

namespace alg {

template<typename T>
Image<T> to_grayscale(const Image<T>& input) {

    if (input.channels() < 3) {
        throw std::runtime_error("Grayscale requires at least 3 channels");
    }

    Image<T> gray(input.width(), input.height(), 1);

    const float maxVal = PixelTraits<T>::max();

    for (int y = 0; y < input.height(); ++y) {
        for (int x = 0; x < input.width(); ++x) {

            // normalize to [0,1]
            float r = static_cast<float>(input.at(x, y, 0)) / maxVal;
            float g = static_cast<float>(input.at(x, y, 1)) / maxVal;
            float b = static_cast<float>(input.at(x, y, 2)) / maxVal;

            // grayscale in float
            float grayVal = 0.299f * r + 0.587f * g + 0.114f * b;

            // scale back to native range
            float outVal = grayVal * maxVal;
            outVal = std::clamp(outVal,
                                PixelTraits<T>::min(),
                                PixelTraits<T>::max());

            gray.at(x, y, 0) = static_cast<T>(outVal);
        }
    }

    return gray;
}
}
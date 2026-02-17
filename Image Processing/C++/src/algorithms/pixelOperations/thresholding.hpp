#pragma once

#include "../../image/image.hpp"
#include "grayscale.hpp"
#include "../helpers/histogram.hpp"
#include "../helpers/otsu.hpp"
#include "../../image/pixel_traits.hpp"

#include <algorithm>
#include <stdexcept>

namespace alg {

// Forward declaration
template<typename T>
Image<T> otsuThresholding(const Image<T>& input);

template<typename T>
Image<T> thresholding(const Image<T>& input, float threshold_norm)
{
    // Ensure grayscale
    Image<T> gray = input;
    if (input.channels() > 1) {
        gray = to_grayscale(input);
    }

    // Automatic mode → Otsu
    if (threshold_norm < 0.0f) {
        return otsuThresholding(gray);
    }

    threshold_norm = std::clamp(threshold_norm, 0.0f, 1.0f);

    const float maxVal = PixelTraits<T>::max();
    const float Tval   = threshold_norm * maxVal;

    Image<T> output(gray.width(), gray.height(), 1);

    for (int y = 0; y < gray.height(); ++y) {
        for (int x = 0; x < gray.width(); ++x) {

            float v = static_cast<float>(gray.at(x, y, 0));

            output.at(x, y, 0) =
                (v < Tval)
                    ? static_cast<T>(PixelTraits<T>::min())
                    : static_cast<T>(v);
        }
    }

    return output;
}

template<typename T>
Image<T> otsuThresholding(const Image<T>& input)
{
    if (input.channels() != 1) {
        throw std::runtime_error(
            "Otsu thresholding requires a grayscale image"
        );
    }

    Histogram hist(input);

    const int Tbin = Otsu::threshold(
        hist.channel(0),
        hist.pixelCount()
    );

    const float maxVal = PixelTraits<T>::max();
    const float Tval =
        (static_cast<float>(Tbin) / (hist.bins() - 1)) * maxVal;

    Image<T> output(input.width(), input.height(), 1);

    for (int y = 0; y < input.height(); ++y) {
        for (int x = 0; x < input.width(); ++x) {

            float v = static_cast<float>(input.at(x, y, 0));

            output.at(x, y, 0) =
                (v < Tval)
                    ? static_cast<T>(PixelTraits<T>::min())
                    : static_cast<T>(v);
        }
    }

    return output;
}

} // namespace alg

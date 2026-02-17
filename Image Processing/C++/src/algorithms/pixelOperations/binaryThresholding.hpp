#pragma once
#include "../../image/image.hpp"
#include "../../image/pixel_traits.hpp"
#include <stdexcept>
#include <algorithm>

namespace alg {

template<typename T>
Image<T> binaryThresholding(const Image<T>& input, float threshold) {

    if (input.channels() != 1) {
        throw std::runtime_error(
            "Binary thresholding requires a grayscale image"
        );
    }

    threshold = std::clamp(threshold, 0.0f, 1.0f);

    Image<T> output(
        input.width(),
        input.height(),
        1
    );

    const float maxVal = PixelTraits<T>::max();
    const float minVal = PixelTraits<T>::min();

    for (size_t i = 0; i < input.data().size(); ++i) {

        float vNorm =
            static_cast<float>(input.data()[i]) / maxVal;

        output.data()[i] =
            (vNorm >= threshold)
                ? static_cast<T>(maxVal)
                : static_cast<T>(minVal);
    }

    return output;
}
}

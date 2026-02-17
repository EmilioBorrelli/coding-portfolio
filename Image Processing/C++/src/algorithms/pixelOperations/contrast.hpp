#pragma once
#include "../../image/image.hpp"
#include "../../image/pixel_traits.hpp"
#include <stdexcept>
#include <algorithm>

namespace alg{

template<typename T>
Image<T> contrast(const Image<T>& input, float alpha) {

    if (alpha <= 0.0f) {
        throw std::invalid_argument("Contrast alpha must be > 0");
    }

    Image<T> output(
        input.width(),
        input.height(),
        input.channels()
    );

    const float maxVal = PixelTraits<T>::max();
    constexpr float mid = 0.5f;

    for (size_t i = 0; i < input.data().size(); ++i) {

        // normalize
        float v = static_cast<float>(input.data()[i]) / maxVal;

        // contrast around mid-gray
        v = (v - mid) * alpha + mid;

        // clamp & denormalize
        float outVal = std::clamp(v * maxVal,
                            PixelTraits<T>::min(),
                            PixelTraits<T>::max());
        output.data()[i] = static_cast<T>(outVal);
    }

    return output;
}
}
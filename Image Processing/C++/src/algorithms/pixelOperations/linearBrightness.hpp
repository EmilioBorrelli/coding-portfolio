#pragma once
#include "../../image/image.hpp"
#include "../../image/pixel_traits.hpp"

#include <algorithm>

namespace alg {

template<typename T>
Image<T> linearBrightness(const Image<T>& input, float beta_norm) {

    Image<T> output(
        input.width(),
        input.height(),
        input.channels()
    );

    const float maxVal = PixelTraits<T>::max();
    const float minVal = PixelTraits<T>::min();

    // Convert normalized beta to native range
    const float beta = beta_norm * maxVal;

    for (size_t i = 0; i < input.data().size(); ++i) {
        float v = static_cast<float>(input.data()[i]) + beta;
        v = std::clamp(v, minVal, maxVal);
        output.data()[i] = static_cast<T>(v);
    }

    return output;
}

} // namespace alg

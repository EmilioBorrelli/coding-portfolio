#pragma once

#include "../../image/image.hpp"

#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace alg {

template<typename T>
Image<float> gradientMagnitude(
    const Image<T>& gx,
    const Image<T>& gy)
{
    if (gx.width()  != gy.width() ||
        gx.height() != gy.height() ||
        gx.channels() != gy.channels())
    {
        throw std::invalid_argument(
            "gradientMagnitude: image size mismatch"
        );
    }

    Image<float> out(
        gx.width(),
        gx.height(),
        gx.channels()
    );

    for (size_t i = 0; i < gx.data().size(); ++i) {

        float dx = static_cast<float>(gx.data()[i]);
        float dy = static_cast<float>(gy.data()[i]);

        out.data()[i] = std::sqrt(dx * dx + dy * dy);
    }

    return out;
}

} // namespace alg


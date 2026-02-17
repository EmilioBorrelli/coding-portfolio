#pragma once

#include "../../image/image.hpp"
#include <algorithm>
#include <stdexcept>

namespace alg {

enum class EdgeClass : uint8_t {
    None   = 0,
    Weak   = 1,
    Strong = 2
};

inline Image<uint8_t> doubleThreshold(
    const Image<float>& magnitude,
    float lowRatio,
    float highRatio)
{
    if (lowRatio < 0.0f || highRatio > 1.0f || lowRatio >= highRatio)
        throw std::invalid_argument("Invalid threshold ratios");

    Image<uint8_t> result(
        magnitude.width(),
        magnitude.height(),
        1
    );

    const auto& data = magnitude.data();
    float maxVal = *std::max_element(data.begin(), data.end());

    float highT = maxVal * highRatio;
    float lowT  = maxVal * lowRatio;

    for (size_t i = 0; i < data.size(); ++i)
    {
        float v = data[i];

        if (v >= highT)
            result.data()[i] = 255;     // strong edge
        else if (v >= lowT)
            result.data()[i] = 100;     // weak edge (gray)
        else
            result.data()[i] = 0;       // no edge
    }

    return result;
}

}

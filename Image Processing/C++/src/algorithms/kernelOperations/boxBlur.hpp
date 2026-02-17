#pragma once

#include "../../image/image.hpp"
#include "../helpers/convolution.hpp"

#include <vector>
#include <stdexcept>

namespace alg {

template<typename T>
Image<float> boxBlur(
    const Image<T>& input,
    int kernelSize,
    BorderType border = BorderType::Clamp)
{
    if (kernelSize <= 0 || kernelSize % 2 == 0)
        throw std::invalid_argument("boxBlur: kernelSize must be positive and odd");

    const int k = kernelSize;
    const float norm = 1.0f / (k * k);

    std::vector<float> kernel(k * k, norm);

    return convolve(input, kernel, k, k, border);
}

} // namespace alg


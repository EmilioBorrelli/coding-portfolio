#pragma once

#include "../../image/image.hpp"
#include "../helpers/convolution.hpp"
#include "../pixelOperations/grayscale.hpp"

#include <vector>
#include <stdexcept>
#include <utility>

namespace alg {

template<typename T>
std::pair<Image<float>, Image<float>> prewitt(
    const Image<T>& input,
    BorderType border = BorderType::Clamp)
{
    Image<T> gray = input;
    if (gray.channels() > 1) {
        gray = alg::to_grayscale(input);
    }

    constexpr int k = 3;

    const std::vector<float> kernelX = {
        -1, 0, 1,
        -1, 0, 1,
        -1, 0, 1
    };

    const std::vector<float> kernelY = {
        -1, -1, -1,
         0,  0,  0,
         1,  1,  1
    };

    Image<float> gx = convolve(gray, kernelX, k, k, border);
    Image<float> gy = convolve(gray, kernelY, k, k, border);

    return { gx, gy };
}

} // namespace alg

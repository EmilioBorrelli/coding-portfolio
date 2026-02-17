#pragma once

#include "../../image/image.hpp"
#include "../helpers/convolution.hpp"
#include "../pixelOperations/grayscale.hpp"

#include <vector>
#include <stdexcept>

namespace alg {

enum class LaplacianType {
    FourNeighbor,
    EightNeighbor
};

template<typename T>
Image<float> laplacian(
    const Image<T>& input,
    LaplacianType variant = LaplacianType::FourNeighbor,
    BorderType border = BorderType::Clamp)
{
    Image<T> gray = input;
    if (gray.channels() > 1) {
        gray = alg::to_grayscale(input);
    }

    if (gray.channels() != 1)
        throw std::invalid_argument("Laplacian requires single-channel image");

    constexpr int k = 3;
    std::vector<float> kernel;

    if (variant == LaplacianType::FourNeighbor) {
        kernel = {
             0, -1,  0,
            -1,  4, -1,
             0, -1,  0
        };
    }
    else {
        kernel = {
            -1, -1, -1,
            -1,  8, -1,
            -1, -1, -1
        };
    }

    return convolve(gray, kernel, k, k, border);
}

} // namespace alg

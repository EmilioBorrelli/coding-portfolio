#pragma once

#include "../../image/image.hpp"

// Kernel ops
#include "../kernelOperations/gaussianBlur.hpp"
#include "../kernelOperations/laplacian.hpp"
#include "../kernelOperations/zeroCrossing.hpp"

namespace alg {

template<typename T>
Image<float> laplacianOfGaussian(
    const Image<T>& input,
    float sigma = 1.0f,
    BorderType border = BorderType::Clamp)
{
    // Convert to float
    Image<float> working = input.toFloat();
    if (working.channels() > 1)
        working = to_grayscale(working);

    // Gaussian blur
    sigma = std::max(sigma, 0.5f);
    int radius = static_cast<int>(std::ceil(3 * sigma));
    int k = 2 * radius + 1;
    Image<float> blurred = gaussianBlur(working, k, sigma, border);

    // Laplacian
    Image<float> lap = laplacian(blurred);

    for (auto& v : lap.data())
        v *= sigma * sigma;


    // Zero crossing
    Image<float> edges = zeroCrossing(lap, 0.09f, border);

    return edges;
}

}

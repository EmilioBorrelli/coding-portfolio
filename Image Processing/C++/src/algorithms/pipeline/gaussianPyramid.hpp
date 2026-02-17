#pragma once

#include "../../image/image.hpp"
#include "../kernelOperations/gaussianBlur.hpp"
#include "../pixelOperations/grayscale.hpp"
#include "../helpers/downsample.hpp"

#include <vector>
#include <cmath>

namespace alg {

template<typename T>
std::vector<Image<float>> buildGaussianPyramid(
    const Image<T>& input,
    int levels,
    float sigma = 1.6f,
    BorderType border = BorderType::Clamp)
{
    std::vector<Image<float>> pyramid;

    // Convert to float + grayscale
    Image<float> current = input.toFloat();
    if (current.channels() > 1)
        current = to_grayscale(current);

    pyramid.push_back(current);

    for (int i = 1; i < levels; ++i)
    {
        int radius = static_cast<int>(std::ceil(3 * sigma));
        int k = 2 * radius + 1;

        Image<float> blurred =
            gaussianBlur(current, k, sigma, border);

        Image<float> reduced =
            downsample(blurred);

        pyramid.push_back(reduced);

        current = reduced;
    }

    return pyramid;
}

}

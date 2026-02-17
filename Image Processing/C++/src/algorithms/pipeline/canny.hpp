#pragma once

#include "../../image/image.hpp"

// Pixel ops
#include "../pixelOperations/grayscale.hpp"

// Kernel ops
#include "../kernelOperations/gaussianBlur.hpp"
#include "../kernelOperations/sobel.hpp"

// Helpers
#include "../helpers/gradientMagnitude.hpp"
#include "../helpers/gradientDirection.hpp"
#include "../helpers/nonMaxSuppression.hpp"
#include "../helpers/doubleThreshold.hpp"
#include "../helpers/hysteresis.hpp"

namespace alg {

template<typename T>
Image<uint8_t> canny(
    const Image<T>& input,
    float lowRatio = 0.05f,
    float highRatio = 0.15f,
    int gaussianKernel = 5,
    float sigma = 1.0f)
{
    // 1️⃣ Grayscale
    Image<T> gray = input;
    if (gray.channels() > 1)
        gray = to_grayscale(input);

    // 2️⃣ Promote to float
    Image<float> grayF = gray.toFloat();

    // 3️⃣ Gaussian smoothing
    Image<float> smooth = gaussianBlur(grayF, gaussianKernel, sigma);

    // 4️⃣ Sobel
    auto [gx, gy] = sobel(smooth);

    // 5️⃣ Gradient magnitude
    Image<float> mag = gradientMagnitude(gx, gy);

    // 6️⃣ Gradient direction
    Image<uint8_t> dir = computeGradientDirection(gx, gy);

    // 7️⃣ Non-max suppression
    Image<float> thin = nonMaxSuppression(mag, dir);

    // 8️⃣ Double threshold
    // Scale thresholds
    Image<uint8_t> dt = doubleThreshold(thin, lowRatio, highRatio);


    // 9️⃣ Hysteresis
    auto& d = dt.data();
    auto [minIt, maxIt] = std::minmax_element(d.begin(), d.end());
    std::cout << "DoubleThreshold min=" << int(*minIt)
            << " max=" << int(*maxIt) << std::endl;
    Image<uint8_t> edges = hysteresis(dt);
    auto& e = edges.data();
    auto [minIt2, maxIt2] = std::minmax_element(e.begin(), e.end());
    std::cout << "Hysteresis min=" << int(*minIt2)
            << " max=" << int(*maxIt2) << std::endl;

    return edges;
}

}

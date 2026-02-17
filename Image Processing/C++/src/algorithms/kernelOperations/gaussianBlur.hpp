#pragma once

#include "../../image/image.hpp"
#include "../helpers/convolution.hpp"

#include <vector>
#include <stdexcept>
#include <cmath>

namespace alg {

    template<typename T>
    Image<float> gaussianBlur(
        const Image<T>& input,
        int kernelSize,
        float sigma,
        BorderType border = BorderType::Clamp
    ){
        if (kernelSize <= 0 || kernelSize % 2 == 0)
            throw std::invalid_argument(
                "gaussianBlur: kernelSize must be positive and odd"
            );

        if (sigma <= 0.0f)
            throw std::invalid_argument(
                "gaussianBlur: sigma must be > 0"
            );

        const int k = kernelSize;
        const int half = k / 2;

        std::vector<float> kernel(k * k);
        float sum = 0.0f;

        for (int j = 0; j < k; ++j) {
            for (int i = 0; i < k; ++i) {
                int x = i - half;
                int y = j - half;

                float value =
                    std::exp(-(x * x + y * y) / (2.0f * sigma * sigma));

                kernel[j * k + i] = value;
                sum += value;
            }
        }

        for (float& v : kernel)
            v /= sum;

        return convolve(input, kernel, k, k, border);
    }

}

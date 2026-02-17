#pragma once

#include "../../image/image.hpp"
#include <vector>

#include <algorithm>
#include <cassert>
#include <stdexcept>




namespace alg {

enum class BorderType {
    Zero,
    Clamp,
    Mirror
};

namespace detail {

// Handle border according to policy
inline int handleBorder(int p, int max, BorderType border)
{
    switch (border) {
        case BorderType::Zero:
            return -1; // signal invalid
        case BorderType::Clamp:
            return std::clamp(p, 0, max - 1);
        case BorderType::Mirror:
            if (p < 0)      return -p - 1;
            if (p >= max)   return 2 * max - p - 1;
            return p;
    }
    return -1;
}

} // namespace detail

template<typename T>
Image<float> convolve(
    const Image<T>& input,
    const std::vector<float>& kernel,
    int kernelWidth,
    int kernelHeight,
    BorderType border = BorderType::Clamp)
{
    if (kernelWidth <= 0 || kernelHeight <= 0)
        throw std::invalid_argument("Kernel dimensions must be > 0");

    if (kernelWidth * kernelHeight != static_cast<int>(kernel.size()))
        throw std::invalid_argument("Kernel size mismatch");

    if (kernelWidth % 2 == 0 || kernelHeight % 2 == 0)
        throw std::invalid_argument("Kernel dimensions must be odd");

    Image<float> output(
        input.width(),
        input.height(),
        input.channels()
    );

    const int kHalfW = kernelWidth  / 2;
    const int kHalfH = kernelHeight / 2;

    for (int y = 0; y < input.height(); ++y) {
        for (int x = 0; x < input.width(); ++x) {
            for (int c = 0; c < input.channels(); ++c) {

                float sum = 0.0f;

                for (int ky = 0; ky < kernelHeight; ++ky) {
                    for (int kx = 0; kx < kernelWidth; ++kx) {

                        int ix = x + kx - kHalfW;
                        int iy = y + ky - kHalfH;

                        ix = detail::handleBorder(ix, input.width(),  border);
                        iy = detail::handleBorder(iy, input.height(), border);

                        if (ix < 0 || iy < 0)
                            continue; // zero padding

                        float pixel =
                            static_cast<float>(input.at(ix, iy, c));

                        float weight =
                            kernel[ky * kernelWidth + kx];

                        sum += pixel * weight;
                    }
                }

                output.at(x, y, c) = sum;
            }
        }
    }

    return output;
}

} // namespace alg




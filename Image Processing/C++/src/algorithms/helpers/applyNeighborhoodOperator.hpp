#pragma once

#include "../../image/image.hpp"
#include <vector>
#include <stdexcept>
#include <algorithm>
#include "convolution.hpp"

namespace alg {

template<typename T, typename Operator>
Image<T> applyNeighborhoodOperator(
    const Image<T>& input,
    int kernelWidth,
    int kernelHeight,
    Operator op,
    BorderType border = BorderType::Clamp)
{
    if (kernelWidth <= 0 || kernelHeight <= 0)
        throw std::invalid_argument("Kernel size must be > 0");

    if (kernelWidth % 2 == 0 || kernelHeight % 2 == 0)
        throw std::invalid_argument("Kernel size must be odd");

    Image<T> output(
        input.width(),
        input.height(),
        input.channels()
    );

    const int kHalfW = kernelWidth  / 2;
    const int kHalfH = kernelHeight / 2;

    std::vector<T> neighborhood;
    neighborhood.reserve(kernelWidth * kernelHeight);

    for (int y = 0; y < input.height(); ++y) {
        for (int x = 0; x < input.width(); ++x) {
            for (int c = 0; c < input.channels(); ++c) {

                neighborhood.clear();

                for (int ky = -kHalfH; ky <= kHalfH; ++ky) {
                    for (int kx = -kHalfW; kx <= kHalfW; ++kx) {

                        int ix = x + kx;
                        int iy = y + ky;

                        ix = detail::handleBorder(ix, input.width(), border);
                        iy = detail::handleBorder(iy, input.height(), border);

                        if (ix < 0 || iy < 0)
                            continue;

                        neighborhood.push_back(
                            input.at(ix, iy, c)
                        );
                    }
                }

                output.at(x, y, c) = op(neighborhood);
            }
        }
    }

    return output;
}

} // namespace alg

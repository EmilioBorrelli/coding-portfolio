#pragma once

#include "../../image/image.hpp"

#include <stdexcept>
#include <algorithm>

namespace alg {

inline Image<float> nonMaxSuppression(
    const Image<float>& mag,
    const Image<uint8_t>& dir)
{
    if (mag.width()  != dir.width() ||
        mag.height() != dir.height())
    {
        throw std::invalid_argument(
            "nonMaxSuppression: image size mismatch"
        );
    }

    Image<float> out(
        mag.width(),
        mag.height(),
        1
    );

    // Initialize output to zero
    std::fill(out.data().begin(), out.data().end(), 0.0f);

    // Skip borders (need neighbors)
    for (int y = 1; y < mag.height() - 1; ++y) {
        for (int x = 1; x < mag.width() - 1; ++x) {

            float m = mag.at(x, y, 0);
            uint8_t d = dir.at(x, y, 0);

            float m1 = 0.0f;
            float m2 = 0.0f;

            switch (d) {
                case 0: // 0° (left-right)
                    m1 = mag.at(x - 1, y, 0);
                    m2 = mag.at(x + 1, y, 0);
                    break;

                case 1: // 45° (top-right / bottom-left)
                    m1 = mag.at(x + 1, y - 1, 0);
                    m2 = mag.at(x - 1, y + 1, 0);
                    break;

                case 2: // 90° (top-bottom)
                    m1 = mag.at(x, y - 1, 0);
                    m2 = mag.at(x, y + 1, 0);
                    break;

                case 3: // 135° (top-left / bottom-right)
                    m1 = mag.at(x - 1, y - 1, 0);
                    m2 = mag.at(x + 1, y + 1, 0);
                    break;

                default:
                    continue; // invalid direction
            }

            if (m >= m1 && m >= m2) {
                out.at(x, y, 0) = m;
            }
            // else remains zero
        }
    }

    return out;
}

} // namespace alg


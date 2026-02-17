#pragma once

#include "../helpers/applyNeighborhoodOperator.hpp"
#include <algorithm>
#include <cmath>

namespace alg {

Image<float> zeroCrossing(
    const Image<float>& input,
    float relativeFactor = 0.03f,
    BorderType border = BorderType::Clamp)
{
    constexpr int kernelSize = 3;

    // Compute max absolute response
    float maxAbs = 0.0f;

    for (float v : input.data())
        maxAbs = std::max(maxAbs, std::abs(v));

    float threshold = relativeFactor * maxAbs;

    return applyNeighborhoodOperator<float>(
        input,
        kernelSize,
        kernelSize,
        [threshold](const std::vector<float>& values) -> float
        {
            bool hasPos = false;
            bool hasNeg = false;

            float minVal = values[0];
            float maxVal = values[0];

            for (float v : values)
            {
                if (v > 0) hasPos = true;
                if (v < 0) hasNeg = true;

                minVal = std::min(minVal, v);
                maxVal = std::max(maxVal, v);
            }

            const int center = values.size() / 2;
            float c = values[center];

            for (int i = 0; i < values.size(); ++i)
            {
                if (i == center) continue;

                float n = values[i];

                if ((c > 0 && n < 0) ||
                    (c < 0 && n > 0))
                {
                    if (std::abs(c - n) > threshold)
                        return 1.0f;
                }
            }
            return 0.0f;
        },
        border
    );
}

}

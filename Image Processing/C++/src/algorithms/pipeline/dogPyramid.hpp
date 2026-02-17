#pragma once

#include "../../image/image.hpp"
#include <vector>

namespace alg {

std::vector<Image<float>> buildDoGPyramid(
    const std::vector<Image<float>>& gaussianPyramid)
{
    std::vector<Image<float>> dog;

    for (size_t i = 1; i < gaussianPyramid.size(); ++i)
    {
        const auto& Gprev = gaussianPyramid[i - 1];
        const auto& Gcurr = gaussianPyramid[i];

        Image<float> diff(
            Gcurr.width(),
            Gcurr.height(),
            Gcurr.channels()
        );

        for (int y = 0; y < Gcurr.height(); ++y)
        {
            for (int x = 0; x < Gcurr.width(); ++x)
            {
                diff.at(x, y, 0) =
                    Gcurr.at(x, y, 0)
                  - Gprev.at(x * 2, y * 2, 0);
            }
        }

        dog.push_back(diff);
    }

    return dog;
}

}

#pragma once

#include "../../image/image.hpp"

#include <vector>
#include <cmath>

namespace alg {

struct SIFTKeypoint
{
    int octave;
    int scale;
    int x;
    int y;
    float response;
};

inline std::vector<SIFTKeypoint> detectScaleSpaceExtrema(
    const std::vector<std::vector<Image<float>>>& dog,
    float contrastThreshold = 0.03f)
{
    std::vector<SIFTKeypoint> keypoints;

    for (size_t o = 0; o < dog.size(); ++o)
    {
        for (size_t s = 1; s < dog[o].size() - 1; ++s)
        {
            const auto& prev = dog[o][s - 1];
            const auto& curr = dog[o][s];
            const auto& next = dog[o][s + 1];

            for (int y = 1; y < curr.height() - 1; ++y)
            {
                for (int x = 1; x < curr.width() - 1; ++x)
                {
                    float v = curr.at(x, y, 0);

                    if (std::abs(v) < contrastThreshold)
                        continue;

                    bool isMax = true;
                    bool isMin = true;

                    for (int ds = -1; ds <= 1; ++ds)
                    {
                        const Image<float>& img =
                            (ds == -1) ? prev :
                            (ds ==  0) ? curr :
                                         next;

                        for (int dy = -1; dy <= 1; ++dy)
                        {
                            for (int dx = -1; dx <= 1; ++dx)
                            {
                                if (ds == 0 && dy == 0 && dx == 0)
                                    continue;

                                float neighbor =
                                    img.at(x + dx, y + dy, 0);

                                if (v <= neighbor)
                                    isMax = false;

                                if (v >= neighbor)
                                    isMin = false;

                                if (!isMax && !isMin)
                                    goto done_check;
                            }
                        }
                    }

                done_check:

                    if (isMax || isMin)
                    {
                        keypoints.push_back({
                            static_cast<int>(o),
                            static_cast<int>(s),
                            x,
                            y,
                            v
                        });
                    }
                }
            }
        }
    }

    return keypoints;
}

} // namespace alg

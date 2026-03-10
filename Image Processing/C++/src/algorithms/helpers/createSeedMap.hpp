#pragma once

#include "../../image/image.hpp"
#include "../pipeline/siftExtremaDetection.hpp"

#include <vector>
#include <algorithm>
#include <cassert>

namespace alg {

inline Image<uint8_t> createSeedMap(
    const std::vector<SIFTKeypoint>& keypoints,
    int width,
    int height)
{
    Image<uint8_t> seeds(width, height, 1);

    assert(seeds.channels() == 1);

    std::fill(seeds.data().begin(),
              seeds.data().end(),
              0);

    for (const auto& kp : keypoints)
    {
        int scaleFactor = 1 << kp.octave;

        int x = kp.x * scaleFactor;
        int y = kp.y * scaleFactor;

        if (x >= 0 && x < width &&
            y >= 0 && y < height)
        {
            seeds.at(x,y,0) = 255;
        }
    }

    return seeds;
}

}
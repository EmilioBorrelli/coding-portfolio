#pragma once

#include "../../image/image.hpp"
#include "siftExtremaDetection.hpp"
#include "siftGaussianPyramid.hpp"
#include <vector>
#include <cmath>

namespace alg {

struct OrientedKeypoint : SIFTKeypoint
{
    float orientation;
};

inline std::vector<OrientedKeypoint> assignSIFTOrientation(
    const std::vector<SIFTKeypoint>& keypoints,
    const SIFTPyramid& pyramid,
    int bins = 36)
{
    const auto& gaussianPyramid = pyramid.octaves;
    std::vector<OrientedKeypoint> result;
    const float M_PI = 3.141592653589793; 
    for (const auto& kp : keypoints)
    {
        const auto& img =
            gaussianPyramid[kp.octave][kp.scale];

        std::vector<float> hist(bins, 0.0f);

        int radius = 8;

        for (int dy = -radius; dy <= radius; ++dy)
        {
            for (int dx = -radius; dx <= radius; ++dx)
            {
                int x = kp.x + dx;
                int y = kp.y + dy;

                if (x <= 0 || x >= img.width()-1 ||
                    y <= 0 || y >= img.height()-1)
                    continue;

                float gx =
                    img.at(x+1,y,0)
                  - img.at(x-1,y,0);

                float gy =
                    img.at(x,y+1,0)
                  - img.at(x,y-1,0);

                float mag = std::sqrt(gx*gx + gy*gy);
                float angle = std::atan2(gy,gx);

                int bin =
                    static_cast<int>(
                        bins * (angle + M_PI) / (2*M_PI)
                    ) % bins;

                hist[bin] += mag;
            }
        }

        int maxBin = std::distance(
            hist.begin(),
            std::max_element(hist.begin(), hist.end())
        );

        float orientation =
            2 * M_PI * maxBin / bins - M_PI;

        result.push_back(
            {kp.octave, kp.scale, kp.x, kp.y, kp.response, orientation}
        );
    }

    return result;
}

} // namespace alg

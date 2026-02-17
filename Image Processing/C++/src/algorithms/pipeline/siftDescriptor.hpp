#pragma once

#include "../../image/image.hpp"
#include "siftOrientation.hpp"
#include "siftGaussianPyramid.hpp"

#include <vector>
#include <cmath>

namespace alg {

struct SIFTDescriptor
{
    OrientedKeypoint keypoint;
    std::vector<float> descriptor; // 128D
};

inline std::vector<SIFTDescriptor> computeSIFTDescriptors(
    const std::vector<OrientedKeypoint>& keypoints,
    const SIFTPyramid& pyramid)
{
    const auto& gaussianPyramid = pyramid.octaves;
    std::vector<SIFTDescriptor> result;
    const float M_PI = 3.141592653589793; 
    const int gridSize = 4;
    const int bins = 8;
    const int radius = 8;

    for (const auto& kp : keypoints)
    {
        const auto& img =
            gaussianPyramid[kp.octave][kp.scale];

        std::vector<float> descriptor(
            gridSize * gridSize * bins, 0.0f);

        for (int dy = -radius; dy < radius; ++dy)
        {
            for (int dx = -radius; dx < radius; ++dx)
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
                float angle =
                    std::atan2(gy,gx) - kp.orientation;

                while (angle < 0) angle += 2*M_PI;
                while (angle >= 2*M_PI) angle -= 2*M_PI;

                int cellX = (dx + radius) / (radius/2);
                int cellY = (dy + radius) / (radius/2);

                if (cellX < 0 || cellX >= gridSize ||
                    cellY < 0 || cellY >= gridSize)
                    continue;

                int bin =
                    static_cast<int>(
                        bins * angle / (2*M_PI)
                    ) % bins;

                int idx =
                    (cellY * gridSize + cellX) * bins + bin;

                descriptor[idx] += mag;
            }
        }

        // L2 normalize
        float norm = 0.0f;
        for (float v : descriptor)
            norm += v * v;
        norm = std::sqrt(norm);

        if (norm > 0)
            for (float& v : descriptor)
                v /= norm;

        result.push_back({kp, descriptor});
    }

    return result;
}

} // namespace alg

#pragma once

#include "../../image/image.hpp"
#include "siftExtremaDetection.hpp"

#include <vector>
#include <cmath>

namespace alg {

    inline std::vector<SIFTKeypoint> eliminateEdgeResponses(
    const std::vector<SIFTKeypoint>& keypoints,
    const std::vector<std::vector<Image<float>>>& dog,
    float edgeRatio = 10.0f)
    {

        std::vector<SIFTKeypoint> filtered;
        float edgeThreshold = (edgeRatio + 1.0f) * (edgeRatio + 1.0f) / edgeRatio;
        for (const auto& kp : keypoints)
        {
            const Image<float>& img =
                dog[kp.octave][kp.scale];

            int x = kp.x;
            int y = kp.y;

            // Skip if too close to border
            if (x <= 0 || x >= img.width() - 1 ||
                y <= 0 || y >= img.height() - 1)
                continue;

            // Second derivatives (finite differences)

            float Dxx =
                img.at(x+1,y,0)
            - 2.0f * img.at(x,y,0)
            + img.at(x-1,y,0);

            float Dyy =
                img.at(x,y+1,0)
            - 2.0f * img.at(x,y,0)
            + img.at(x,y-1,0);

            float Dxy =
                ( img.at(x+1,y+1,0)
                - img.at(x+1,y-1,0)
                - img.at(x-1,y+1,0)
                + img.at(x-1,y-1,0) )
                * 0.25f;

            float trace = Dxx + Dyy;
            float det   = Dxx * Dyy - Dxy * Dxy;

            // Reject saddle points or unstable points
            if (det <= 0.0f)
                continue;

            float curvatureRatio = (trace * trace) / det;

            if (curvatureRatio < edgeThreshold)
            {
                filtered.push_back(kp);
            }
        }

    return filtered;
}


}
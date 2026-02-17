#pragma once
#include "../../image/image.hpp"
#include "siftGaussianPyramid.hpp"

namespace alg {

std::vector<std::vector<Image<float>>> buildSIFTDoG(
    const SIFTPyramid& gp)
{
    std::vector<std::vector<Image<float>>> dog;

    for (const auto& octave : gp.octaves)
    {
        std::vector<Image<float>> dogOctave;

        for (size_t i = 1; i < octave.size(); ++i)
        {
            const auto& prev = octave[i - 1];
            const auto& curr = octave[i];

            Image<float> diff(
                curr.width(),
                curr.height(),
                1
            );

            for (int y = 0; y < curr.height(); ++y)
                for (int x = 0; x < curr.width(); ++x)
                    diff.at(x,y,0) =
                        curr.at(x,y,0)
                      - prev.at(x,y,0);

            dogOctave.push_back(diff);
        }

        dog.push_back(dogOctave);
    }

    return dog;
}

}

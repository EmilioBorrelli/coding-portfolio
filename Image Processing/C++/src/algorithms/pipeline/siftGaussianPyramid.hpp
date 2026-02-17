#pragma once

#include "../../image/image.hpp"
#include "../kernelOperations/gaussianBlur.hpp"
#include "../pixelOperations/grayscale.hpp"
#include "../helpers/downsample.hpp"

#include <vector>
#include <cmath>

namespace alg {

struct SIFTPyramid
{
    std::vector<std::vector<Image<float>>> octaves;
};

template<typename T>
SIFTPyramid buildSIFTPyramid(
    const Image<T>& input,
    int numOctaves = 4,
    int scalesPerOctave = 3,
    float sigma0 = 1.6f,
    BorderType border = BorderType::Clamp)
{
    SIFTPyramid pyramid;

    float k = std::pow(2.0f, 1.0f / scalesPerOctave);

    // Convert input to grayscale float
    Image<float> base = input.toFloat();
    if (base.channels() > 1)
        base = to_grayscale(base);

    for (int o = 0; o < numOctaves; ++o)
    {
        std::vector<Image<float>> scales;

        Image<float> current = base;

        float sigmaPrev = sigma0;

        scales.push_back(current);

        for (int s = 1; s < scalesPerOctave + 3; ++s)
        {
            float sigmaTotal = sigma0 * std::pow(k, s);
            float sigmaInc = std::sqrt(
                sigmaTotal * sigmaTotal
              - sigmaPrev * sigmaPrev
            );

            int radius = static_cast<int>(std::ceil(3 * sigmaInc));
            int ksize = 2 * radius + 1;

            Image<float> blurred =
                gaussianBlur(current, ksize, sigmaInc, border);

            scales.push_back(blurred);

            current = blurred;
            sigmaPrev = sigmaTotal;
        }

        pyramid.octaves.push_back(scales);

        // Downsample for next octave
        base = downsample(scales[scalesPerOctave]);
    }

    return pyramid;
}

}

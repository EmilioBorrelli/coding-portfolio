#pragma once
#include "../../image/image.hpp"
#include "../helpers/histogram.hpp"
#include "../../image/pixel_traits.hpp"

#include <vector>
#include <algorithm>

namespace alg {

template<typename T>
Image<T> histogramEqualisation(const Image<T>& input)
{
    // Handle empty image
    if (input.width() == 0 || input.height() == 0 || input.channels() == 0)
        return input;

    Histogram<T> hist(input);

    Image<T> output(
        input.width(),
        input.height(),
        input.channels()
    );

    const std::size_t bins   = hist.bins();
    const std::size_t pixels = hist.pixelCount();
    const float maxVal       = PixelTraits<T>::max();

    // --- build CDF per channel ---
    std::vector<std::vector<float>> cdf(input.channels(),
                                        std::vector<float>(bins, 0.0f));

    for (std::size_t c = 0; c < input.channels(); ++c) {

        const auto& h = hist.channel(c);

        cdf[c][0] = static_cast<float>(h[0]);
        for (std::size_t i = 1; i < bins; ++i) {
            cdf[c][i] = cdf[c][i - 1] + static_cast<float>(h[i]);
        }

        // normalize to [0,1]
        for (float& v : cdf[c]) {
            v /= static_cast<float>(pixels);
        }
    }

    // --- apply equalisation ---
    for (size_t i = 0; i < input.data().size(); ++i) {

        const std::size_t c = i % input.channels();

        float v = static_cast<float>(input.data()[i]);
        std::size_t idx = static_cast<std::size_t>(
            (v / maxVal) * (bins - 1)
        );

        idx = std::min(idx, bins - 1);

        float outVal = cdf[c][idx] * maxVal;
        outVal = std::clamp(outVal,
                            PixelTraits<T>::min(),
                            PixelTraits<T>::max());

        output.data()[i] = static_cast<T>(outVal);
    }

    return output;
}

} // namespace alg

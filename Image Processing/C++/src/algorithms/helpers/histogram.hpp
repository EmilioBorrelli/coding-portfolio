#pragma once

#include "../../image/image.hpp"
#include "../../image/pixel_traits.hpp"

#include <vector>
#include <cstddef>
#include <algorithm>
#include <stdexcept>

template<typename T>
class Histogram {
public:
    using BinType = std::uint32_t;

    explicit Histogram(const Image<T>& img)
        : m_bins(PixelTraits<T>::bins()),
          m_hist(img.channels(),
                 std::vector<BinType>(m_bins, 0)),
          m_pixelCount(0)
    {
        if (img.width() == 0 || img.height() == 0 || img.channels() == 0)
            throw std::runtime_error("Histogram: empty image");

        const float minVal = PixelTraits<T>::min();
        const float maxVal = PixelTraits<T>::max();

        m_pixelCount =
            static_cast<std::size_t>(img.width()) *
            static_cast<std::size_t>(img.height());

        for (int y = 0; y < img.height(); ++y) {
            for (int x = 0; x < img.width(); ++x) {
                for (int c = 0; c < img.channels(); ++c) {

                    float v =
                        (static_cast<float>(img.at(x, y, c)) - minVal) /
                        (maxVal - minVal);

                    v = std::clamp(v, 0.0f, 1.0f);

                    std::size_t bin =
                        static_cast<std::size_t>(v * (m_bins - 1));

                    m_hist[c][bin]++;
                }
            }
        }
    }

    std::size_t bins() const { return m_bins; }
    std::size_t channels() const { return m_hist.size(); }
    std::size_t pixelCount() const { return m_pixelCount; }

    const std::vector<BinType>& channel(std::size_t c) const {
        if (c >= m_hist.size())
            throw std::out_of_range("Histogram channel out of range");
        return m_hist[c];
    }

private:
    std::size_t m_bins;
    std::vector<std::vector<BinType>> m_hist;
    std::size_t m_pixelCount;
};

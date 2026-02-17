#pragma once
#include "../../image/image.hpp"

#include "histogramEqualisation.hpp"
#include "../helpers/histogram.hpp"
#include "../../image/pixel_traits.hpp"
#include <vector>
#include <array>
#include <cstdint>
#include <algorithm>

namespace alg {

template<typename T>
Image<T> CLAHE(const Image<T>& input,
               int tileWidth,
               int tileHeight,
               float clipLimit)
{
    if (input.width() == 0 || input.height() == 0 || input.channels() == 0)
    return input;

    if (tileWidth <= 0 || tileHeight <= 0)
        return input;

    if (clipLimit <= 0.0f)
        return input;

    Image<T> output(input.width(), input.height(), input.channels());

    const int tilesX = (input.width()  + tileWidth  - 1) / tileWidth;
    const int tilesY = (input.height() + tileHeight - 1) / tileHeight;

    const float maxVal = PixelTraits<T>::max();

    // --------------------------------------------------
    // Phase 1: build clipped CDF per tile & channel
    // --------------------------------------------------
    struct TileCDF {
        std::vector<std::vector<float>> cdf; // [channel][bin]
    };

    std::vector<std::vector<TileCDF>> tiles(
        tilesY, std::vector<TileCDF>(tilesX)
    );

    for (int ty = 0; ty < tilesY; ++ty) {
        for (int tx = 0; tx < tilesX; ++tx) {

            int x0 = tx * tileWidth;
            int y0 = ty * tileHeight;
            int x1 = std::min(x0 + tileWidth,  input.width());
            int y1 = std::min(y0 + tileHeight, input.height());

            Image<T> tile(x1 - x0, y1 - y0, input.channels());

            for (int y = y0; y < y1; ++y)
                for (int x = x0; x < x1; ++x)
                    for (int c = 0; c < input.channels(); ++c)
                        tile.at(x - x0, y - y0, c) = input.at(x, y, c);

            Histogram<T> hist(tile);

            const std::size_t bins   = hist.bins();
            const std::size_t pixels = hist.pixelCount();

            tiles[ty][tx].cdf.resize(
                input.channels(),
                std::vector<float>(bins, 0.0f)
            );

            for (std::size_t c = 0; c < input.channels(); ++c) {

                std::vector<uint32_t> h = hist.channel(c);

                // --- clip histogram ---
                float clip = clipLimit * pixels;
                
                    
                float excess = 0.0f;

                for (auto& v : h) {
                    if (v > clip) {
                        excess += (v - clip);
                        v = static_cast<uint32_t>(clip);
                    }
                }

                // redistribute excess uniformly
                uint32_t redist = static_cast<uint32_t>(excess / bins);
                uint32_t rem    = static_cast<uint32_t>(excess) % bins;

                for (auto& v : h)
                    v += redist;

                for (std::size_t i = 0; i < rem; ++i)
                    h[i]++;

                // --- build CDF ---
                tiles[ty][tx].cdf[c][0] = h[0];
                for (std::size_t i = 1; i < bins; ++i)
                    tiles[ty][tx].cdf[c][i] =
                        tiles[ty][tx].cdf[c][i - 1] + h[i];

                for (auto& v : tiles[ty][tx].cdf[c])
                    v /= static_cast<float>(pixels);
            }
        }
    }

    // --------------------------------------------------
    // Phase 2: bilinear interpolation
    // --------------------------------------------------
    for (int y = 0; y < input.height(); ++y) {
        for (int x = 0; x < input.width(); ++x) {

            int tx = x / tileWidth;
            int ty = y / tileHeight;

            int tx1 = std::min(tx + 1, tilesX - 1);
            int ty1 = std::min(ty + 1, tilesY - 1);

            float fx = (x - tx * tileWidth) / float(tileWidth);
            float fy = (y - ty * tileHeight) / float(tileHeight);

            for (int c = 0; c < input.channels(); ++c) {

                float v = static_cast<float>(input.at(x, y, c)) / maxVal;
                v = std::clamp(v, 0.0f, 1.0f);
                
                std::size_t bins = tiles[ty][tx].cdf[c].size();
                std::size_t bin = std::min(
                    static_cast<std::size_t>(v * (bins - 1)),
                    bins - 1
                );

                float v00 = tiles[ty ][tx ].cdf[c][bin];
                float v10 = tiles[ty ][tx1].cdf[c][bin];
                float v01 = tiles[ty1][tx ].cdf[c][bin];
                float v11 = tiles[ty1][tx1].cdf[c][bin];

                float top    = v00 * (1 - fx) + v10 * fx;
                float bottom = v01 * (1 - fx) + v11 * fx;
                float outVal = top * (1 - fy) + bottom * fy;

                output.at(x, y, c) =
                    static_cast<T>(
                        std::clamp(outVal * maxVal,
                        PixelTraits<T>::min(),
                        PixelTraits<T>::max())
                    );
            }
        }
    }

    return output;
}

} // namespace alg
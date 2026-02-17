#pragma once
#include "../../image/image.hpp"

namespace alg {

template<typename T>
Image<T> downsample(const Image<T>& input)
{
    int newW = input.width() / 2;
    int newH = input.height() / 2;

    Image<T> out(newW, newH, input.channels());

    for (int y = 0; y < newH; ++y)
    {
        for (int x = 0; x < newW; ++x)
        {
            for (int c = 0; c < input.channels(); ++c)
            {
                out.at(x, y, c) =
                    input.at(x * 2, y * 2, c);
            }
        }
    }

    return out;
}

}

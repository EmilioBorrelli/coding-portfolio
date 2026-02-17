#pragma once

#include "../../image/image.hpp"
#include <stack>
#include <utility>

namespace alg {

inline Image<uint8_t> hysteresis(
    const Image<uint8_t>& classified)
{
    Image<uint8_t> result = classified;

    const int w = result.width();
    const int h = result.height();

    std::stack<std::pair<int,int>> stack;

    constexpr uint8_t NONE   = 0;
    constexpr uint8_t WEAK   = 128;
    constexpr uint8_t STRONG = 255;

    // --------------------------------------------------
    // 1. Push all strong edges
    // --------------------------------------------------
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            if (result.at(x,y,0) == STRONG)
                stack.push({x,y});

    // --------------------------------------------------
    // 2. DFS propagation
    // --------------------------------------------------
    while (!stack.empty())
    {
        auto [x,y] = stack.top();
        stack.pop();

        for (int dy = -1; dy <= 1; ++dy)
        {
            for (int dx = -1; dx <= 1; ++dx)
            {
                if (dx == 0 && dy == 0)
                    continue;

                int nx = x + dx;
                int ny = y + dy;

                if (nx < 0 || ny < 0 || nx >= w || ny >= h)
                    continue;

                if (result.at(nx,ny,0) == WEAK)
                {
                    result.at(nx,ny,0) = STRONG;
                    stack.push({nx,ny});
                }
            }
        }
    }

    // --------------------------------------------------
    // 3. Remove non-strong pixels
    // --------------------------------------------------
    for (size_t i = 0; i < result.data().size(); ++i)
    {
        if (result.data()[i] != STRONG)
            result.data()[i] = 0;
    }

    return result;
}

} // namespace alg

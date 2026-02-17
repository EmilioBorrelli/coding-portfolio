#pragma once

#include "../helpers/applyNeighborhoodOperator.hpp"
#include <algorithm>

namespace alg {

template<typename T>
Image<T> medianFilter(
    const Image<T>& input,
    int kernelSize,
    BorderType border = BorderType::Clamp)
{
    return applyNeighborhoodOperator<T>(
        input,
        kernelSize,
        kernelSize,
        [](std::vector<T>& values) -> T {

            std::sort(values.begin(), values.end());
            return values[values.size() / 2];
        },
        border
    );
}

} // namespace alg

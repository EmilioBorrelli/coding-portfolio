#pragma once

#include "../helpers/applyNeighborhoodOperator.hpp"
#include <algorithm>

namespace alg {

template<typename T>
Image<T> erosion(
    const Image<T>& input,
    int kernelSize,
    BorderType border = BorderType::Clamp)
{
    return applyNeighborhoodOperator<T>(
        input,
        kernelSize,
        kernelSize,
        [](std::vector<T>& values) -> T {
            return *std::min_element(values.begin(), values.end());
        },
        border
    );
}

} // namespace alg

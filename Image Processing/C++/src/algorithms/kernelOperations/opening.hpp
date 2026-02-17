#pragma once

#include "../helpers/applyNeighborhoodOperator.hpp"
#include <algorithm>
#include "erosion.hpp"
#include "dilation.hpp"

namespace alg {

template<typename T>
Image<T> opening(
    const Image<T>& input,
    int kernelSize,
    BorderType border = BorderType::Clamp)
{
    Image<T> erosion = alg::erosion(input,kernelSize,border);
    Image<T> opening = alg::dilation(erosion,kernelSize,border);
    return opening;
}

} // namespace alg

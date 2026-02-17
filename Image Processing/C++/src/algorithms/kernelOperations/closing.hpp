#pragma once

#include "../helpers/applyNeighborhoodOperator.hpp"
#include <algorithm>
#include "erosion.hpp"
#include "dilation.hpp"

namespace alg {

template<typename T>
Image<T> closing(
    const Image<T>& input,
    int kernelSize,
    BorderType border = BorderType::Clamp)
{
    Image<T> dilated = alg::dilation(input,kernelSize,border);
    Image<T> opening = alg::erosion(dilated,kernelSize,border);
    
    return opening;
}
}

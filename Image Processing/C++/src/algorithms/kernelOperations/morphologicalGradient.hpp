#pragma once

#include "../helpers/applyNeighborhoodOperator.hpp"
#include <algorithm>
#include "erosion.hpp"
#include "dilation.hpp"

namespace alg {

template<typename T>
Image<float> morphologicalGradient(
    const Image<T>& input,
    int kernelSize,
    BorderType border = BorderType::Clamp)
{
    Image<T> dilated = alg::dilation(input, kernelSize, border);
    Image<T> eroded  = alg::erosion(input, kernelSize, border);

    Image<float> output(
        input.width(),
        input.height(),
        input.channels()
    );

    for (size_t i = 0; i < output.data().size(); ++i) {
        output.data()[i] =
            static_cast<float>(dilated.data()[i]) -
            static_cast<float>(eroded.data()[i]);
    }

    return output;
}

}

#pragma once
#include "../../image/image.hpp"
#include "../../image/pixel_traits.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace alg{

    template<typename T>
    Image<T> gammaCorrection(const Image<T>& input, float gamma){
        if (gamma <= 0.0f){
            throw std::invalid_argument("Gamma must be > 0");
        }
        // create an empty image with same dimensions as input image
        Image<T> correctedImage(input.width(), input.height(), input.channels());
        // iterate through every input image pixel and chanel
        const float maxVal = PixelTraits<T>::max();

        for (size_t i = 0; i < input.data().size(); ++i) {

            float iN = static_cast<float>(input.data()[i]) / maxVal;
            iN = std::clamp(iN, 0.0f, 1.0f);

            float corrected = std::pow(iN, gamma) * maxVal;
            corrected = std::clamp(corrected,
                       PixelTraits<T>::min(),
                       PixelTraits<T>::max());

            correctedImage.data()[i] = static_cast<T>(corrected);
        }
        return correctedImage;    
    }
}
#pragma once

#include "../../image/image.hpp"

#include <cmath>
#include <stdexcept>
#include <numbers>
#include <cstdint>


namespace alg {

/**
 * Quantized gradient directions used for Non-Maximum Suppression
 */
enum class GradientDirection : uint8_t {
    Deg0   = 0,  // 0 degrees   (horizontal)
    Deg45  = 1,  // 45 degrees
    Deg90  = 2,  // 90 degrees  (vertical)
    Deg135 = 3   // 135 degrees
};

/**
 * Quantize gradient direction given Gx and Gy.
 * Input: gradient components (can be signed).
 * Output: one of 4 canonical directions.
 */
inline GradientDirection
quantizeGradientDirection(float gx, float gy)
{
    const float M_PI = 3.141592653589793;

    // atan2 returns angle in radians [-pi, pi]
    float angle = std::atan2(gy, gx) * 180.0f / static_cast<float>(M_PI);

    // Map to [0, 180)
    if (angle < 0.0f)
        angle += 180.0f;

    if (angle < 22.5f || angle >= 157.5f)
        return GradientDirection::Deg0;
    else if (angle < 67.5f)
        return GradientDirection::Deg45;
    else if (angle < 112.5f)
        return GradientDirection::Deg90;
    else
        return GradientDirection::Deg135;
}

/**
 * Compute quantized gradient direction image from Gx and Gy.
 * Output image stores direction indices:
 *   0 -> Deg0
 *   1 -> Deg45
 *   2 -> Deg90
 *   3 -> Deg135
 */
template<typename T>
Image<uint8_t> computeGradientDirection(
    const Image<T>& gx,
    const Image<T>& gy)
{
    if (gx.width()  != gy.width() ||
        gx.height() != gy.height() ||
        gx.channels() != gy.channels())
    {
        throw std::invalid_argument(
            "computeGradientDirection: image size mismatch"
        );
    }
    if (gx.channels() != 1)
        throw std::invalid_argument("GradientDirection requires single-channel images");

    Image<uint8_t> dir(
        gx.width(),
        gx.height(),
        1
    );

    for (size_t i = 0; i < gx.data().size(); ++i) {

        float dx = static_cast<float>(gx.data()[i]);
        float dy = static_cast<float>(gy.data()[i]);

        dir.data()[i] =
            static_cast<uint8_t>(
                quantizeGradientDirection(dx, dy)
            );
    }

    return dir;
}

} // namespace alg


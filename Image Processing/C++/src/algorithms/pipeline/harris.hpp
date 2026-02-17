#pragma once

#include "../../image/image.hpp"
#include "../kernelOperations/gaussianBlur.hpp"
#include "../kernelOperations/sobel.hpp"
#include "../pixelOperations/grayscale.hpp"

#include <vector>
#include <cmath>

namespace alg {

struct HarrisKeypoint
{
    int x;
    int y;
    float response;
};

template<typename T>
std::vector<HarrisKeypoint> harris(
    const Image<T>& input,
    float k = 0.04f,
    float threshold = 0.01f,
    float sigma = 1.0f,
    BorderType border = BorderType::Clamp)
{
    Image<float> img = input.toFloat();
    if (img.channels() > 1)
        img = to_grayscale(img);

    // Compute gradients
    auto [Ix, Iy] = sobel(img);

    int w = img.width();
    int h = img.height();

    Image<float> Ixx(w, h, 1);
    Image<float> Iyy(w, h, 1);
    Image<float> Ixy(w, h, 1);

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            float gx = Ix.at(x,y,0);
            float gy = Iy.at(x,y,0);

            Ixx.at(x,y,0) = gx * gx;
            Iyy.at(x,y,0) = gy * gy;
            Ixy.at(x,y,0) = gx * gy;
        }
    }

    // Gaussian smoothing of structure tensor
    int radius = static_cast<int>(std::ceil(3 * sigma));
    int ksize = 2 * radius + 1;

    Ixx = gaussianBlur(Ixx, ksize, sigma, border);
    Iyy = gaussianBlur(Iyy, ksize, sigma, border);
    Ixy = gaussianBlur(Ixy, ksize, sigma, border);

    std::vector<HarrisKeypoint> keypoints;

    for (int y = 1; y < h - 1; ++y)
    {
        for (int x = 1; x < w - 1; ++x)
        {
            float a = Ixx.at(x,y,0);
            float b = Ixy.at(x,y,0);
            float c = Iyy.at(x,y,0);

            float det = a * c - b * b;
            float trace = a + c;

            float R = det - k * trace * trace;

            if (R > threshold)
            {
                keypoints.push_back({x,y,R});
            }
        }
    }

    return keypoints;
}

} // namespace alg

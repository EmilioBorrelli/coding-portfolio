#include "image/image.hpp"

// Pixel operations
#include "algorithms/pixelOperations/grayscale.hpp"
#include "algorithms/pixelOperations/thresholding.hpp"

// Kernel operations
#include "algorithms/kernelOperations/gaussianBlur.hpp"
#include "algorithms/kernelOperations/sobel.hpp"
#include "algorithms/kernelOperations/laplacian.hpp"

// Helpers
#include "algorithms/helpers/gradientMagnitude.hpp"
#include "algorithms/helpers/gradientDirection.hpp"
#include "algorithms/helpers/nonMaxSuppression.hpp"
#include "algorithms/helpers/doubleThreshold.hpp"
#include "algorithms/helpers/hysteresis.hpp"

// Pipelines
#include "algorithms/pipeline/canny.hpp"
#include "algorithms/pipeline/laplacianOfGaussian.hpp"
#include "algorithms/pipeline/siftExtremaDetection.hpp"
#include "algorithms/pipeline/edgeResponseElimination.hpp"
#include "algorithms/pipeline/siftOrientation.hpp"
#include "algorithms/pipeline/siftDescriptor.hpp"
#include "algorithms/pipeline/harris.hpp"
#include "algorithms/pipeline/siftGaussianPyramid.hpp"
#include "algorithms/pipeline/siftDogPyramid.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

// --------------------------------------------------
// OpenCV conversion
// --------------------------------------------------

template<typename T>
cv::Mat toCvMat(const Image<T>& img) {
    int type;

    if constexpr (std::is_same_v<T, uint8_t>)
        type = CV_8UC(img.channels());
    else if constexpr (std::is_same_v<T, uint16_t>)
        type = CV_16UC(img.channels());
    else if constexpr (std::is_same_v<T, float>)
        type = CV_32FC(img.channels());
    else
        static_assert(false, "Unsupported image type");

    return cv::Mat(
        img.height(),
        img.width(),
        type,
        const_cast<T*>(img.data().data())
    );
}

cv::Mat normalizeFloat(const Image<float>& img) {
    cv::Mat m = toCvMat(img);
    cv::Mat out;
    cv::normalize(m, out, 0, 1, cv::NORM_MINMAX);
    return out;
}

// --------------------------------------------------
// DEMO FUNCTIONS
// --------------------------------------------------

void runCannyDemo(const Image<float>& img) {

    std::cout << "Running Canny Demo...\n";

    auto edges = alg::canny(img, 0.05f, 0.15f, 5, 1.0f);

    cv::imshow("Canny Edges", toCvMat(edges));
    cv::waitKey(0);
}

void runSIFTDemo(const Image<float>& img) {

    std::cout << "Running SIFT Demo...\n";

    auto siftGP = alg::buildSIFTPyramid(img, 4, 3);
    auto siftDoG = alg::buildSIFTDoG(siftGP);

    auto keypoints = alg::detectScaleSpaceExtrema(siftDoG, 0.03f);
    auto filtered = alg::eliminateEdgeResponses(keypoints, siftDoG, 10.0f);
    auto oriented = alg::assignSIFTOrientation(filtered, siftGP);
    auto descriptors = alg::computeSIFTDescriptors(oriented, siftGP);

    std::cout << "Keypoints: " << oriented.size() << "\n";
    std::cout << "Descriptors: " << descriptors.size() << "\n";

    cv::Mat vis = toCvMat(img).clone();

    for (const auto& kp : oriented) {

        int scaleFactor = 1 << kp.octave;
        int x = kp.x * scaleFactor;
        int y = kp.y * scaleFactor;

        float angle = kp.orientation;
        int len = 10;

        int x2 = x + static_cast<int>(len * std::cos(angle));
        int y2 = y + static_cast<int>(len * std::sin(angle));

        cv::circle(vis, {x,y}, 3, {0,255,0}, 1);
        cv::line(vis, {x,y}, {x2,y2}, {0,255,0}, 1);
    }

    cv::imshow("SIFT Keypoints", vis);
    cv::waitKey(0);
}

void runHarrisDemo(const Image<float>& img) {

    std::cout << "Running Harris Demo...\n";

    auto pts = alg::harris(img, 0.04f, 0.005f, 1.0f);

    cv::Mat vis = toCvMat(img).clone();

    for (const auto& kp : pts)
        cv::circle(vis, {kp.x, kp.y}, 2, {255,0,0}, 1);

    cv::imshow("Harris Corners", vis);
    cv::waitKey(0);
}

// --------------------------------------------------
// MAIN
// --------------------------------------------------

int main(int argc, char** argv)
{
    std::string imagePath = "assets/rubberwhale.png";

    if (argc > 1)
        imagePath = argv[1];

    Image<float> img;
    if (!img.load(imagePath)) {
        std::cerr << "Failed to load image: " << imagePath << "\n";
        return 1;
    }

    cv::imshow("Original", toCvMat(img));
    cv::waitKey(0);

    runCannyDemo(img);
    runSIFTDemo(img);
    runHarrisDemo(img);

    return 0;
}

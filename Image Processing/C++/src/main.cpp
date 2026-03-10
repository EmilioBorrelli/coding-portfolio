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
#include "algorithms/helpers/createSeedMap.hpp"

#include "algorithms/pipeline/canny.hpp"
#include "algorithms/pipeline/laplacianOfGaussian.hpp"
#include "algorithms/pipeline/siftExtremaDetection.hpp"
#include "algorithms/pipeline/edgeResponseElimination.hpp"
#include "algorithms/pipeline/siftOrientation.hpp"
#include "algorithms/pipeline/siftDescriptor.hpp"
#include "algorithms/pipeline/harris.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>

#include "algorithms/pipeline/gaussianPyramid.hpp"
#include "algorithms/pipeline/dogPyramid.hpp"
#include "algorithms/pipeline/siftGaussianPyramid.hpp"
#include "algorithms/pipeline/siftDogPyramid.hpp"
// --------------------------------------------------
// OpenCV helpers
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

cv::Mat toCvMatNormalized(const Image<float>& img)
{
    cv::Mat m(img.height(), img.width(),
              CV_32FC(img.channels()),
              const_cast<float*>(img.data().data()));

    double minVal, maxVal;
    cv::minMaxLoc(m, &minVal, &maxVal);

    cv::Mat out;
    m.convertTo(out, CV_32F,
                1.0 / (maxVal - minVal),
                -minVal / (maxVal - minVal));
    return out;
}

void showScaled(
    const std::string& name,
    const cv::Mat& img,
    double scale = 0.5)
{
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(), scale, scale);
    cv::imshow(name, resized);
}
// --------------------------------------------------
// MAIN
// --------------------------------------------------

int main()
{
    // --------------------------------------------------
    // Load
    // --------------------------------------------------
    Image<float> imgF;
    if (!imgF.load("../assets/rubberwhale.png")) {
        std::cerr << "Failed to load image\n";
        return 1;
    }

    showScaled("Original", toCvMat(imgF));
    cv::waitKey(0);

    // --------------------------------------------------
    // Grayscale → float
    // --------------------------------------------------
    Image<float> grayF = alg::to_grayscale(imgF);
    // Image<float> grayF = grayF.toFloat();

    // --------------------------------------------------
    // Gaussian blur
    // --------------------------------------------------
    // Image<float> smooth = alg::gaussianBlur(grayF, 5, 1.0f);
    // showScaled("Gaussian Blur", toCvMatNormalized(smooth));
    // cv::waitKey(0);

    // // --------------------------------------------------
    // // Sobel
    // // --------------------------------------------------
    // auto [gx, gy] = alg::sobel(smooth);

    // showScaled("Sobel Gx", toCvMatNormalized(gx));
    // showScaled("Sobel Gy", toCvMatNormalized(gy));
    // cv::waitKey(0);

    // // --------------------------------------------------
    // // Gradient magnitude
    // // --------------------------------------------------
    // Image<float> mag = alg::gradientMagnitude(gx, gy);
    // showScaled("Gradient Magnitude", toCvMatNormalized(mag));
    // cv::waitKey(0);

    // // --------------------------------------------------
    // // Gradient direction (label image)
    // // --------------------------------------------------
    // Image<uint8_t> dir = alg::computeGradientDirection(gx, gy);

    // // --------------------------------------------------
    // // FIX 3: Color-coded direction visualization
    // // --------------------------------------------------
    // cv::Mat dirColor(dir.height(), dir.width(), CV_8UC3);

    // for (int y = 0; y < dir.height(); ++y) {
    //     for (int x = 0; x < dir.width(); ++x) {
    //         uint8_t d = dir.at(x, y, 0);
    //         cv::Vec3b color;

    //         switch (d) {
    //             case 0: color = {255,   0,   0}; break; // 0°   → Red
    //             case 1: color = {  0, 255,   0}; break; // 45°  → Green
    //             case 2: color = {  0,   0, 255}; break; // 90°  → Blue
    //             case 3: color = {255, 255,   0}; break; // 135° → Yellow
    //             default: color = {0, 0, 0}; break;
    //         }

    //         dirColor.at<cv::Vec3b>(y, x) = color;
    //     }
    // }

    // showScaled("Gradient Direction (color coded)", dirColor);
    // cv::waitKey(0);

    // // --------------------------------------------------
    // // Non-maximum suppression
    // // --------------------------------------------------
    // Image<float> thin = alg::nonMaxSuppression(mag, dir);
    // showScaled("Non-Max Suppression", toCvMatNormalized(thin));
    // cv::waitKey(0);

    // // --------------------------------------------------
    // // Sobel + Otsu edge detection
    // // --------------------------------------------------
    // Image<float> edges = alg::thresholding(mag, -1.0f);
    // showScaled("Sobel + Otsu Edges", toCvMatNormalized(edges));
    // cv::waitKey(0);

    

    // // --------------------------------------------------
    // // Double Threshold
    // // --------------------------------------------------
    // auto [minIt, maxIt] =
    // std::minmax_element(thin.data().begin(), thin.data().end());

    // std::cout << "Thin min=" << *minIt
    //       << " max=" << *maxIt << std::endl;
    // Image<uint8_t> dt =
    //     alg::doubleThreshold(thin, 0.05f, 0.15f);

    // showScaled("Double Threshold", toCvMat(dt));
    // cv::waitKey(0);

    // // --------------------------------------------------
    // // Hysteresis
    // // --------------------------------------------------
    
    // Image<uint8_t> hyst =
    //     alg::hysteresis(dt);

    // showScaled("Hysteresis Result", toCvMat(hyst));
    // cv::waitKey(0);

    // // --------------------------------------------------
    // // Full Canny Wrapper
    // // --------------------------------------------------
    // Image<uint8_t> cannyEdges =
    //     alg::canny(imgF, 0.05f, 0.15f, 5, 1.0f);

    // showScaled("Full Canny", toCvMat(cannyEdges));
    // cv::waitKey(0);

    // Image<float> blurred = alg::gaussianBlur(grayF,7,1.0f);
    // showScaled("Gaussian", toCvMat(blurred));
    // cv::waitKey(0);
    // Image<float> log = alg::laplacian(blurred);
    // showScaled("laplacian of Gaussian", toCvMat(log));
    // cv::waitKey(0);
    // Image<float> lOGEdges = alg::laplacianOfGaussian(imgF,1.9f);
    // showScaled("laplacian of Gaussian pipeline", toCvMat(lOGEdges));
    // cv::waitKey(0);

    // auto gp = alg::buildGaussianPyramid(imgF, 5, 1.6f);
    // auto dog = alg::buildDoGPyramid(gp);

    
    
    // for (size_t o = 0; o < siftGP.octaves.size(); ++o)
    // {
    //     for (size_t s = 0; s < siftGP.octaves[o].size(); ++s)
    //     {
    //         std::string name =
    //             "Gaussian O" + std::to_string(o)
    //             + " S" + std::to_string(s);

    //         cv::Mat m = toCvMatNormalized(
    //             siftGP.octaves[o][s]
    //         );

    //         showScaled(name, m, 0.6);
    //     }
    // }

    // cv::waitKey(0);


    // for (size_t o = 0; o < siftDoG.size(); ++o)
    // {
    //     for (size_t s = 0; s < siftDoG[o].size(); ++s)
    //     {
    //         std::string name =
    //             "DoG O" + std::to_string(o)
    //             + " S" + std::to_string(s);

    //         cv::Mat m = toCvMatNormalized(
    //             siftDoG[o][s]
    //         );

    //         showScaled(name, m, 0.6);
    //     }
    // }

    // cv::waitKey(0);


    auto siftGP = alg::buildSIFTPyramid(imgF, 4, 3);
    auto siftDoG = alg::buildSIFTDoG(siftGP);

    auto keypoints =
    alg::detectScaleSpaceExtrema(siftDoG, 0.03f);

    std::cout << "Detected keypoints: "
            << keypoints.size()
            << std::endl;

    cv::Mat vis = toCvMat(imgF).clone();

    for (const auto& kp : keypoints)
    {
        int scaleFactor = 1 << kp.octave;

        int x = kp.x * scaleFactor;
        int y = kp.y * scaleFactor;

        cv::circle(vis,
                cv::Point(x, y),
                3,
                cv::Scalar(0, 0, 255),
                1);
    }

    cv::imshow("SIFT Keypoints", vis);
    cv::waitKey(0);

    auto filtered =alg::eliminateEdgeResponses(keypoints, siftDoG, 10.0f);
    cv::Mat vis1 = toCvMat(imgF).clone();

    for (const auto& kp : filtered)
    {
        int scaleFactor = 1 << kp.octave;

        int x = kp.x * scaleFactor;
        int y = kp.y * scaleFactor;

        cv::circle(vis1,
                cv::Point(x, y),
                3,
                cv::Scalar(0, 255, 0),
                1);
    }

    cv::imshow("Filtered SIFT Keypoints", vis1);
    cv::waitKey(0); 

    auto oriented =
    alg::assignSIFTOrientation(filtered, siftGP);

    std::cout << "Oriented keypoints: "
            << oriented.size() << std::endl;

    cv::Mat visOriented = toCvMat(imgF).clone();

    for (const auto& kp : oriented)
    {
        int scaleFactor = 1 << kp.octave;

        int x = kp.x * scaleFactor;
        int y = kp.y * scaleFactor;

        float angle = kp.orientation;

        int len = 10;

        int x2 = x + static_cast<int>(len * std::cos(angle));
        int y2 = y + static_cast<int>(len * std::sin(angle));

        cv::circle(visOriented,
                cv::Point(x, y),
                3,
                cv::Scalar(0,255,0),
                1);

        cv::line(visOriented,
                cv::Point(x,y),
                cv::Point(x2,y2),
                cv::Scalar(0,255,0),
                1);
    }

    cv::imshow("Oriented SIFT Keypoints", visOriented);
    cv::waitKey(0);

    auto descriptors =
    alg::computeSIFTDescriptors(oriented, siftGP);

    std::cout << "Descriptors computed: "
            << descriptors.size() << std::endl;

    cv::waitKey(0);

    auto seeds = alg::createSeedMap(
    filtered,
    imgF.width(),
    imgF.height()
    );
    showScaled("Seed Map", toCvMat(seeds));

    cv::Mat seedVis = cv::Mat::zeros(imgF.height(), imgF.width(), CV_8UC3);

    for (const auto& kp : filtered)
    {
        int scaleFactor = 1 << kp.octave;
        int x = kp.x * scaleFactor;
        int y = kp.y * scaleFactor;

        cv::circle(seedVis, cv::Point(x, y), 3, cv::Scalar(255,255,255), -1);
    }

    showScaled("Seed Map Debug", seedVis);
    cv::waitKey(0);
    
    int seedCount = 0;
    for (auto v : seeds.data())
    {
        if (v > 0) ++seedCount;
    }
    std::cout << "Seed pixels: " << seedCount << std::endl;
    
    cv::waitKey(0);
 
    
    auto harrisPts =
    alg::harris(imgF, 0.04f, 0.005f, 1.0f);

    cv::Mat visHarris = toCvMat(imgF).clone();

    for (const auto& kp : harrisPts)
    {
        cv::circle(visHarris,
                cv::Point(kp.x, kp.y),
                2,
                cv::Scalar(255,0,0),
                1);
    }

    cv::imshow("Harris Corners", visHarris);
    cv::waitKey(0);
    return 0;

}

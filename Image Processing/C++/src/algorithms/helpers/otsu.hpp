#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
class Otsu {
public:
    static std::size_t threshold(
        const std::vector<std::uint32_t>& hist,
        std::size_t pixelCount
    )
    {
        if (pixelCount == 0 || hist.empty()) {
            throw std::runtime_error("Otsu threshold: invalid histogram");
        }

        const std::size_t bins = hist.size();

        std::vector<double> p(bins);
        for (std::size_t i = 0; i < bins; ++i)
            p[i] = static_cast<double>(hist[i]) / pixelCount;

        double muT = 0.0;
        for (std::size_t i = 0; i < bins; ++i)
            muT += i * p[i];

        double omega0 = 0.0;
        double mu0 = 0.0;

        double maxVariance = -1.0;
        std::size_t bestThreshold = 0;

        for (std::size_t T = 0; T < bins; ++T) {
            omega0 += p[T];
            mu0 += T * p[T];

            double omega1 = 1.0 - omega0;
            if (omega0 <= 0.0 || omega1 <= 0.0)
                continue;

            double mu0Mean = mu0 / omega0;
            double mu1 = (muT - mu0) / omega1;

            double varBetween =
                omega0 * omega1 * (mu0Mean - mu1) * (mu0Mean - mu1);

            if (varBetween > maxVariance) {
                maxVariance = varBetween;
                bestThreshold = T;
            }
        }

        return bestThreshold;
    }
};
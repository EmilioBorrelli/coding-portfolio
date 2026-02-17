#pragma once
#include <cstdint>
#include <limits>

template<typename T>
struct PixelTraits;

// ---------------- uint8 ----------------
template<>
struct PixelTraits<uint8_t> {
    static constexpr float min() { return 0.0f; }
    static constexpr float max() { return 255.0f; }
    static constexpr std::size_t bins() { return 256; }
};

// ---------------- uint16 ----------------
template<>
struct PixelTraits<uint16_t> {
    static constexpr float min() { return 0.0f; }
    static constexpr float max() { return 65535.0f; }
    static constexpr std::size_t bins() { return 256; } // IMPORTANT
};

// ---------------- float ----------------
template<>
struct PixelTraits<float> {
    static constexpr float min() { return 0.0f; }
    static constexpr float max() { return 1.0f; }
    static constexpr std::size_t bins() { return 256; }
};

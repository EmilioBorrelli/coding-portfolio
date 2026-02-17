# C++ Computer Vision Engine

A modular computer vision engine implemented from scratch in modern C++17.

This project focuses on building classical computer vision algorithms without relying on high-level OpenCV feature detectors. OpenCV is used only for image I/O and visualization — all core algorithms are implemented manually.

The goal of this project is to demonstrate strong fundamentals in image processing, multi-scale analysis, feature detection, and clean C++ software architecture.

---

## 🚀 Implemented Features

### Image Processing
- Grayscale conversion
- Thresholding (manual + Otsu)
- Gaussian blur
- Laplacian filtering
- Sobel gradients

### Edge Detection
- Gradient magnitude
- Gradient direction quantization
- Non-maximum suppression
- Double threshold
- Hysteresis
- Full Canny edge detection pipeline
- Laplacian of Gaussian (LoG)

### Multi-Scale Analysis
- Gaussian Pyramid
- Difference of Gaussian (DoG)
- SIFT Gaussian Pyramid
- SIFT DoG Pyramid

### Feature Detection
- Scale-space extrema detection
- Edge response elimination (Hessian-based)
- SIFT orientation assignment
- SIFT descriptor generation
- Harris corner detection

---

## 🏗 Architecture

```
cpp_cv_engine/
│
├── assets/              # Demo images
├── examples/            # Demo applications
│   └── demo.cpp
├── src/
│   ├── image/           # Core Image<T> container
│   └── algorithms/      # All algorithms & pipelines
├── external/            # Third-party headers (if needed)
├── CMakeLists.txt
└── README.md
```

The core processing code is compiled into a reusable CMake library target:

```
cv_engine
```

Two executables are built:

- `image_app`
- `demo`

Both link against the same engine library.

---

## 🖼 Demo Application

The demo program showcases:

- Canny edge detection
- SIFT keypoint detection
- SIFT orientation visualization
- SIFT descriptor computation
- Harris corner detection

### Build

```bash
cmake -S . -B build -G Ninja
cmake --build build
```

### Run (from project root)

```bash
.\build\demo.exe
```

### Run with custom image

```bash
.\build\demo.exe assets/starry_night.jpg
```

Demo images are stored in the `assets/` folder.

---

## ⚙ Dependencies

- C++17
- CMake ≥ 3.21
- OpenCV (for visualization and image loading only)

All feature detection and image processing algorithms are implemented manually.

---

## 🧠 Core Design Principles

- Modern C++17
- Template-based `Image<T>` container (`uint8_t`, `uint16_t`, `float`)
- Clean separation between algorithms and pipelines
- Modular CMake structure
- Reusable engine library target
- Minimal external dependencies
- Clear algorithmic structure

---

## 📊 Engineering Highlights

- Multi-octave SIFT implementation
- Scale-space extrema detection
- Hessian-based edge suppression
- Orientation histogram assignment
- Descriptor vector construction
- Separable convolution framework
- Pipeline-based algorithm composition

---

## 🎯 Project Motivation

This project was built to:

- Deepen understanding of classical computer vision
- Implement algorithms mathematically rather than relying on black-box libraries
- Practice clean C++ architecture
- Create a reusable foundation for further CV development

---

## 🚀 Potential Extensions

- SIMD acceleration
- Parallel processing
- Feature matching & homography
- Image stitching
- Benchmarking vs OpenCV implementations
- Real-time pipeline integration

---

## 👤 Author

Emilio Borrelli  
Engineering Portfolio — Computer Vision & Signal Processing

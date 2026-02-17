# Coding Portfolio — Computer Vision & EEG Signal Processing

This repository contains two independent but conceptually aligned engineering projects:

1. A modular C++ computer vision engine implemented from scratch  
2. A research-grade Python EEG preprocessing and feature extraction pipeline  

Both projects emphasize algorithmic understanding, clean architecture, and reproducible engineering.

---

# 📁 Repository Structure

```
coding-portfolio/
│
├── cpp_cv_engine/          # C++ Computer Vision Engine
└── python_eeg_pipeline/    # EEG Preprocessing & Feature Pipeline
```

---

# 🧠 Project 1 — C++ Computer Vision Engine

A modular computer vision framework built in modern C++17.

## Highlights

- Custom templated `Image<T>` container
- Manual implementations of classical CV algorithms
- Multi-scale analysis (Gaussian & DoG pyramids)
- Full Canny edge detection pipeline
- SIFT:
  - Scale-space extrema detection
  - Edge response elimination
  - Orientation assignment
  - Descriptor generation
- Harris corner detection
- Clean CMake library architecture

OpenCV is used only for:
- Image loading
- Visualization

All algorithmic logic is implemented manually.

## Build

```bash
cd cpp_cv_engine
cmake -S . -B build -G Ninja
cmake --build build
```

## Run Demo

From project root:

```bash
.\build\demo.exe
```

Or:

```bash
.\build\demo.exe assets/lena.png
```

---

# 🧠 Project 2 — Python EEG Preprocessing Pipeline

A modular EEG preprocessing and feature extraction framework built with:

- MNE
- NumPy
- SciPy
- Custom preprocessing logic
- Structured quality control reporting

## Implemented Stages

- Line noise evaluation & removal
- Adaptive high-pass selection
- ICA artifact detection (EOG)
- Optional ASR artifact suppression
- Spectral feature extraction
- Relative & absolute band power
- Connectivity metrics
- QC logging and runtime reporting

The project includes a fully reproducible synthetic demonstration notebook.

## Setup

```bash
cd python_eeg_pipeline
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Launch notebook:

```bash
jupyter notebook demo_pipeline.ipynb
```

---

# 🎯 Engineering Focus

Across both projects:

- Algorithmic implementation over black-box usage
- Reproducibility
- Modular architecture
- Clear separation of concerns
- Strong dependency management
- Clean build systems (CMake + Python venv)

---

# 🚀 Skills Demonstrated

- C++17 system-level programming
- Template-based design
- Signal processing
- Multi-scale computer vision
- Feature extraction pipelines
- Scientific Python ecosystem
- Numerical stability considerations
- Structured quality control logging
- Clean project organization

---

# 🧩 Future Directions

- SIMD optimization (C++)
- Parallel processing
- Feature matching & homography
- EEG classification models
- Real dataset benchmarking
- Performance profiling

---

# 👤 Author

Emilio Borrelli  
Engineering Portfolio — Computer Vision & Signal Processing

# EEG Preprocessing & Feature Extraction Pipeline — Demonstration

This notebook demonstrates a modular, research-grade EEG preprocessing and feature extraction pipeline built for reproducible signal processing and structured quality control.

The pipeline is designed to transform raw EEG recordings into clean signals and structured feature outputs suitable for downstream analysis or machine learning workflows.

---

## 🔬 What This Notebook Demonstrates

This demo uses *synthetic but realistic EEG data* to simulate common real-world artifacts and signal characteristics:

### Simulated Neural Activity
- 1/f background (pink noise)
- Posterior-dominant alpha rhythm (~10 Hz)
- Mild beta component (~20 Hz)

### Simulated Artifacts
- 50 Hz line noise
- Slow baseline drift (~0.3 Hz)
- Eye blinks (with dedicated EOG channel)
- Frontal blink leakage into EEG channels
- Muscle bursts

---

## 🏗 Pipeline Stages

The following processing steps are demonstrated:

1. **Line-Noise Evaluation & Removal**
   - Multiple candidate filters evaluated
   - Residual peak scoring
   - Automatic selection of best solution

2. **Adaptive High-Pass Selection**
   - Drift detection
   - Band preservation evaluation
   - Automatic recommended cutoff

3. **ICA-Based Artifact Removal**
   - Decomposition after high-pass filtering
   - Automatic EOG component detection
   - Component exclusion and signal reconstruction

4. **Optional ASR (Artifact Subspace Reconstruction)**
   - Burst suppression
   - Adaptive subspace cleaning

5. **Feature Extraction**
   - Absolute band power (delta, theta, alpha, beta, gamma)
   - Relative band power
   - Spectral parameterization (FOOOF)
   - Connectivity & network metrics (if enabled)

6. **Quality Control (QC) Logging**
   - Runtime per stage
   - Channel counts
   - NaN rate reporting
   - Stability diagnostics

---

## 📊 Design Philosophy

This project emphasizes:

- Deterministic behavior
- Explicit evaluation of filter quality
- No silent failures
- Structured QC reporting
- Clean modular separation between preprocessing and feature extraction
- Reproducible environments

The demo runs fully standalone and does not require external EEG datasets.

---

## ⚙ Engineering Highlights

- Modular architecture (`windexEEG.py` + `EEGFeatures.py`)
- Clean dependency management (`numpy<2.0` for ecosystem stability)
- Automatic artifact detection using EOG channel
- Signal realism through 1/f noise modeling
- Structured logging for pipeline transparency

---

## 🎯 Intended Use

This pipeline structure is suitable for:

- Neurotechnology applications
- Research EEG preprocessing
- Feature engineering for ML models
- Benchmarking preprocessing strategies
- Educational demonstration of EEG artifact handling

---

## 👤 Author

Emilio Borrelli  
Engineering Portfolio — Neuro-Signal Processing & Computer Vision


## 📦 Installation

Create a clean environment:

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt

# register the venv as kernel
python -m ipykernel install --user --name eeg_portfolio --display-name "Python (EEG Portfolio)"


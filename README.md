# 🎵 UrbanSound8K: Comparative Study of MLP and CNN for Urban Audio Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org)
[![Librosa](https://img.shields.io/badge/Librosa-Audio_Analysis-green.svg)](https://librosa.org/)

## 🎯 Project Overview
This project addresses the challenge of environmental noise classification using the **UrbanSound8K** dataset. I implemented an end-to-end machine learning pipeline—from raw audio signal processing to deep learning deployment—comparing traditional Neural Networks (MLP) against Convolutional Neural Networks (CNN) to determine the most effective feature-architecture pair.

---

## 📈 Executive Performance Summary
The project successfully demonstrated that spatial representation of audio (Mel-Spectrograms) combined with 2D-CNNs significantly outperforms 1D statistical feature sets.

| Architecture | Input Feature | Test Accuracy | Improvement |
| :--- | :--- | :--- | :--- |
| **CNN (2D)** | **Mel-Spectrogram** | **~70%** | **🏆 Best** |
| CNN (2D) | MFCCs (Stacked) | ~66% | -4% |
| MLP (Baseline) | MFCC Mean/Var | ~52% | Baseline |

**Key Takeaway:** Moving from a flat MLP to a CNN architecture resulted in a significant increase in accuracy, proving the model's ability to learn temporal-frequency patterns rather than just statistical snapshots.

---

## 🔬 Technical Deep Dive

### 1. Signal Processing & Feature Engineering
Rather than using raw audio, I transformed the data into high-dimensional representations using **Librosa**:
* **MFCCs (40 Coefficients):** Capturing the "shape" of the sound source.
* **Mel-Spectrograms:** Converting audio into the Mel scale to mimic human hearing sensitivity.
* **Data Integrity:** Implemented a strict **10-Fold Cross-Validation** strategy. I ensured no "Data Leakage" by respecting the original dataset's folds, preventing samples from the same recording session from appearing in both training and test sets.

### 2. Model Architectures
* **Baseline MLP:** A 4-layer dense network with Dropout (0.5) and Batch Normalization to establish a performance floor.
* **Optimized CNN:** A 2D Convolutional pipeline utilizing `Conv2D`, `MaxPooling2D`, and `GlobalAveragePooling` to reduce parameter count while maintaining spatial features.

---

## 📊 Visual Insights & Error Analysis

| Model Comparison (Confusion Matrices) | Analysis |
| :--- | :--- |
| <img src="CNN_data/Mel_Spectrogram/CNN_Mels_final_analysis.png" width="450"> | **Key Findings:** The model excels at detecting "Jackhammers" and "Sirens" due to their distinct periodic patterns. However, it faces challenges distinguishing "Street Music" from "Children Playing" due to shared harmonic structures. |

---

## 🛠️ Installation & Usage

### 1. Clone & Setup
```bash
git clone [https://github.com/Pedrooamaroo/UrbanSound8K_Classification.git](https://github.com/Pedrooamaroo/UrbanSound8K_Classification.git)
cd UrbanSound8K_Classification
pip install -r requirements.txt

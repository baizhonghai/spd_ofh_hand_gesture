# Hand Gesture Recognition Project

This document provides instructions for downloading datasets, extracting features, and training the model for hand gesture recognition.

---

## 📂 Dataset Download

The datasets used in this work are publicly available from the official repository:
- **[Hand Gesture Recognition Dataset Repository](https://github.com/example-link)** *(Please replace with the actual link)*

This repository provides detailed download instructions and dataset access links. In this work, we use the following two datasets:
1. **Cambridge Hand Gesture Dataset**
2. **Northwestern University Hand Gesture Dataset**

---

## ⚙️ Feature Extraction

### 1. SPD Feature Extraction
We adopt the MATLAB implementation from **RiemannianCovDs** and adapt it to our dataset structure to compute local **Symmetric Positive Definite (SPD)** covariance matrices.

**Instructions:**
1. Adapt the code to your local directory structure.
2. Run the following command to generate SPD feature representations:
   ```bash
   python extract_spd_feature.py

### 🌀 Optical Flow Feature Extraction

We further extract **Optical Flow Histogram (OFH)** features to capture temporal motion information from the video sequences.

#### Implementation Details
The core implementation is located at:
`feature_extract_object/OpticalFlow.py`

#### Instructions
Run the following script to produce the second type of feature representation:

```bash
python feature_extract_object/OpticalFlow.py
---

## Training and Evaluation

After extracting both SPD and OFH features, you can train and evaluate the model using:

`python main.py`

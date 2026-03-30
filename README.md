Hand Gesture Recognition Project
This document provides instructions for downloading datasets, extracting features, and training the model for hand gesture recognition.

1. Dataset Download
The datasets used in this work are publicly available from the official repository:

Hand Gesture Recognition Dataset Repository (Please insert the actual URL here)

This repository provides detailed download instructions and dataset access links. In this work, we use the following two datasets:

Cambridge Hand Gesture Dataset

Northwestern University Hand Gesture Dataset

2. Feature Extraction
SPD Feature Extraction
We adopt the MATLAB implementation from RiemannianCovDs and adapt it to our dataset structure to compute local Symmetric Positive Definite (SPD) covariance matrices.

Steps:

Adapt the code to your local directory structure.

Run the following command to generate SPD feature representations:

Bash
python extract_spd_feature.py
Optical Flow Feature Extraction
We further extract Optical Flow Histogram (OFH) features to capture temporal movement.

Implementation path: feature_extract_object/OpticalFlow.py

Steps:

Run the script to produce the second type of feature representation:

Bash
python feature_extract_object/OpticalFlow.py
3. Training and Evaluation
After successfully extracting both SPD and OFH features, you can proceed to train and evaluate the model.

Execution:

Bash
python main.py
4. Acknowledgements
We sincerely thank the authors of RiemannianCovDs for providing the original implementation, which served as the basis for our SPD feature extraction.

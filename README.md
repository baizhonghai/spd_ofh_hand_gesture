Dataset Download

The datasets used in this work are publicly available from the official repository:

Hand Gesture Recognition Dataset Repository

This repository provides detailed download instructions and dataset access links.

In this work, we use the following two datasets:

Cambridge Hand Gesture Dataset
Northwestern University Hand Gesture Dataset
Feature Extraction
1. SPD Feature Extraction

We adopt the MATLAB implementation from RiemannianCovDs and adapt it to our dataset structure to compute local Symmetric Positive Definite (SPD) covariance matrices.

After code adaptation, run:

python extract_spd_feature.py

to generate SPD feature representations.

2. Optical Flow Feature Extraction

We further extract Optical Flow Histogram (OFH) features.

The implementation is located at:

feature_extract_object/OpticalFlow.py

Running this script will produce the second type of feature representation.

Training and Evaluation

After extracting both SPD and OFH features, you can train and evaluate the model using:

python main.py
Acknowledgements

We sincerely thank the authors of RiemannianCovDs for providing the original implementation, which we used as the basis for SPD feature extraction.

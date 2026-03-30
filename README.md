Dataset Download

The datasets used in this work can be obtained from the official repository:

https://github.com/Ha0Tang/HandGestureRecognition

This repository provides download instructions and access links for the datasets.

In this work, we only use the following two datasets:

Cambridge Hand Gesture Dataset
Northwestern University Hand Gesture Dataset

Feature Extraction
1. SPD Feature Extraction

We follow the MATLAB implementation provided in RiemannianCovDs and adapt it to our dataset structure in order to compute local SPD matrices.

After adaptation, run:

python extract_spd_feature.py

to generate SPD features.

2. Optical Flow Feature Extraction

We further extract optical flow histogram (OFH) features using:

feature_extract_object/OpticalFlow.py

This step produces the second type of feature representation.

Training and Evaluation

After obtaining both SPD features and OFH features, you can run the main pipeline:

python main.py

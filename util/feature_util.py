import torch
import numpy as np

def save_feature_to_path(X_features, Y, Verify, save_path):
    X_y_verify = {
        'X': X_features,
        'y': Y,
        'verify': Verify
    }
    torch.save(X_y_verify, save_path)


def read_from_feature_file(feature_path):
    feature = torch.load(feature_path, weights_only=False)
    X = feature['X']
    y = feature['y']
    verify = feature['verify']
    return X, y, verify


def feature_combine(list_A, list_B):
    combined_list = [np.concatenate((a, b)) for a, b in zip(list_A, list_B)]
    return combined_list
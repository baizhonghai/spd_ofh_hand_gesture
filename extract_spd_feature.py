import re

from util.file_process import read_matlab_files
from util.spd_util import map2IDS_vectorize
import torch
import numpy as np
from scipy.linalg import logm
from scipy.io import loadmat
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from util.log_setup import logger
from util.param_load import config
from sklearn.metrics import confusion_matrix
from util.analyze_util import print_confusion


def vectorize_and_concatenate(dict_spd):
    dict_vector = {}
    for label, class_set in dict_spd.items():
        dict_vector[label] = []
        for spd_set in class_set:
            vector_list = []
            for spd_path in spd_set:
                spd = loadmat(spd_path)['spd_RCovDs']
                vector_list.append(map2IDS_vectorize(spd))
            dict_vector[label].append(np.concatenate(vector_list))
    return dict_vector


def split_to_X_y(dict_vector):
    X = []
    y = []
    for label, vector_set in dict_vector.items():
        for vector in vector_set:
            X.append(vector)
            y.append(int(label))
    return X, y


def get_verify_name(path):
    parts = path.split('/')
    return parts[7] + parts[8]


def split_to_X_y_verify(dict_spd):
    X = []
    y = []
    verify = []
    for label, class_set in dict_spd.items():
        for spd_set in class_set:
            vector_list = []
            for spd_path in spd_set:
                spd = loadmat(spd_path)['spd_RCovDs']
                vector_list.append(map2IDS_vectorize(spd))
            X.append(np.concatenate(vector_list))
            y.append(int(label))
            verify.append(get_verify_name(spd_path))
    return X, y, verify


def get_path_config(dataset):
    if dataset == 'Cambridge':
        spd_feature_path = './data/cambridge_1_spd_X_y_verify.pt'
        # mat_spd = '/Users/baizhonghai/TP/RiemannianCovDs/Cambridge_Hand_Gesture_Spd/mat_CG'
        mat_spd = '/Users/bzh/Code/doctor/RiemannianCovDs/Cambridge_Hand_Gesture_Spd/mat_CG'

    elif dataset == 'North':
        spd_feature_path = './data/north_1_spd_X_y_verify_original.pt'
        # mat_spd = '/Users/baizhonghai/TP/RiemannianCovDs/Northwestern_Hand_Gesture_Spd/mat_NG'
        # mat_spd = '/Users/baizhonghai/TP/RiemannianCovDs/data_RGB/mat_NG'
        mat_spd = '/Users/bzh/Code/doctor/RiemannianCovDs/NorthWestern_Mat/mat_NG'
    else:
        raise Exception("you should add dataset config, or you type wrong dataset name")
    return spd_feature_path, mat_spd


if __name__ == '__main__':
    dataset = config['common_settings']['data_set']
    spd_feature_path, mat_spd = get_path_config(dataset)
    top_k = int(config['spd']['topK'])
    if not os.path.isfile(spd_feature_path):
        dict_spd = read_matlab_files(mat_spd, re.compile(r'.*mat'))
        X, y, Verify = split_to_X_y_verify(dict_spd)
        X_y = {
            'X': X,
            'y': y,
            'verify': Verify
        }
        torch.save(X_y, spd_feature_path)

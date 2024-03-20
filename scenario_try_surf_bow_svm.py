import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import sklearn
from scipy.io import loadmat
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from util.log_setup import logger
import torch
from util.param_load import config


def read_data(feature_path):
    X = []
    Y = []
    verify = []
    for label in sorted(os.listdir(feature_path)):
        for mat_set in sorted(os.listdir(os.path.join(feature_path, label))):
            feature_set = []
            for mat in sorted(os.listdir(os.path.join(feature_path, label, mat_set))):
                image_feature = loadmat(os.path.join(feature_path, label, mat_set, mat))['feature']
                feature_set.append(image_feature)
            X.append(feature_set)
            Y.append(label)
            verify.append(label + mat_set)
    return X, Y, verify


# Trich xuat dac trung
def extract_sift_features(X):
    image_descriptors = []
    # sift = cv2.xfeatures2d.SIFT_create()
    sift = cv2.SIFT_create()
    for i in range(len(X)):
        kp, des = sift.detectAndCompute(X[i], None)
        image_descriptors.append(des)

    return image_descriptors


def kmeans_bow(all_descriptors, num_clusters):
    bow_dict = []
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, batch_size=100).fit(all_descriptors)
    bow_dict = kmeans.cluster_centers_
    return bow_dict


# Xay dung vecto dac trung tu dict
def create_features_bow(X, kmeans_model):
    bow_representation = []
    for feature_set in X:
        bow_set_represent = []
        for features in feature_set:
            bow_vector = np.zeros(kmeans_model.n_clusters)
            labels = kmeans_model.predict(features)
            for label in labels:
                bow_vector[label] += 1
            bow_set_represent = np.concatenate((bow_set_represent, bow_vector))
        bow_representation.append(bow_set_represent)
    return bow_representation


# 名字瞎起的，目的就是压缩成一个大的ndarray. list of ndarray.
def transfer_descriptor(X):
    all_in_one = []
    for feature_set in X:
        for feature in feature_set:
            list_of_arrays = [row for row in feature]
            all_in_one = all_in_one + list_of_arrays
    return all_in_one


def save(X_features, Y, Verify):
    X_y_verify = {
        'X': X_features,
        'y': Y,
        'verify': Verify
    }
    torch.save(X_y_verify, f'./data/cambridge_surf_bow_X_y_verify.pt')


if __name__ == '__main__':
    X, Y, Verify = read_data('datasets/cambridge_hand_gesture_surf_feature')

    X_all_descriptors = transfer_descriptor(X)

    num_clusters = 64#100

    kmeans_model = MiniBatchKMeans(n_clusters=num_clusters, batch_size=100).fit(X_all_descriptors)

    X_features = create_features_bow(X, kmeans_model)

    if True:
        save(X_features, Y, Verify)

    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y, test_size=float(config['common_settings']['test_size']),
                                                        random_state=42)

    #也可以进行kbest选择下为了一致性


    svm = sklearn.svm.SVC(C=10)
    svm.fit(X_train, Y_train)

    # Accuracy
    accuracy = svm.score(X_test, Y_test)
    logger.info(f'paramers:kmeansCenter:{num_clusters},accuracy:{accuracy}')

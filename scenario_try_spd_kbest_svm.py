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
            y.append(label)
    return X, y


def get_verify_name(path):
    parts = path.split('/')
    return parts[7]+parts[8]

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
            y.append(label)
            verify.append(get_verify_name(spd_path))
    return X, y, verify


if __name__ == '__main__':

    top_k = int(config['spd']['topK'])
    if not os.path.isfile(f'./data/cambridge_1_spd_X_y_verify.pt'):
        dict_spd = read_matlab_files('/Users/baizhonghai/TP/RiemannianCovDs/Cambridge_Hand_Gesture_Spd/mat_CG')
        # vectorize and connect each subset save as one vector

        # dict_vector = vectorize_and_concatenate(dict_spd)
        # X, y = split_to_X_y(dict_vector)
        X, y, Verify = split_to_X_y_verify(dict_spd)
        X_y = {
            'X': X,
            'y': y,
            'verify': Verify
        }
        torch.save(X_y, f'./data/cambridge_1_spd_X_y_verify.pt')
    else:
        X_y = torch.load(f'./data/cambridge_1_spd_X_y_verify.pt')
        X = X_y['X']
        y = X_y['y']
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(config['common_settings']['test_size']), random_state=42)

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Perform PCA
        '''pca = PCA(n_components=0.5)  # You can adjust the explained variance threshold as needed
        X_train_selected = pca.fit_transform(X_train_scaled)
        X_test_selected = pca.transform(X_test_scaled)
        '''
        # bestK
        selector_spd = SelectKBest(mutual_info_classif, k=top_k)
        X_train_selected = torch.tensor(selector_spd.fit_transform(X_train_scaled, y_train))
        X_test_selected = torch.tensor(selector_spd.transform(X_test_scaled))

        # Train SVM classifier
        # svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        # svm = SVC(kernel='linear', C=1.0, gamma='scale', random_state=42)
        svm = SVC(kernel='linear', C=10, gamma='scale', random_state=42)
        svm.fit(X_train_selected, y_train)

        # Make predictions
        y_pred = svm.predict(X_test_selected)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

        # Example usage
        logger.info(f'parameters:topK:{top_k},accuracy:{accuracy}')

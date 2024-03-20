from util.file_process import read_matlab_files
from util.spd_util import map2IDS_vectorize
# 尝试1，直接用log Euclidean 转换成Euclidean域内的东西然后进行bestk or pca降为分类，
# 尝试2，再找个滑动窗口的尺寸，而后构造第二个SPD矩阵组，将1&2的结果融合后bestk. 这里可以增加窗口的尺寸数量，至多4个吧。
# 尝试3，将SPD矩阵丢到SPDNet中去处理。
# 尝试4，将多尺寸的SPD组合成一个大的矩阵，丢到SPDNet处理。
# 尝试5，调整SPDNet并行处理这些不同维度的SPD,而后连接起来分类。
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
import logging
import logging.config

logging.config.fileConfig('./config/logging_config.ini')
logger = logging.getLogger('my_app')

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


def filter(data):
    filtered_data = {}
    import re
    pattern = re.compile(r'.*28M28M.*')  # Compile the regular expression pattern

    for key, value in data.items():
        filtered_items = []
        for sublist in value:
            for item in sublist:
                if pattern.match(item):
                    filtered_items.append([item])
                    break  # Break after finding the first matching item in the sublist
        if filtered_items:  # If there are filtered items, add them to the filtered data
            filtered_data[key] = filtered_items
    return filtered_data


if __name__ == '__main__':
    three_spd = True
    top_k = 100
    if not os.path.isfile(f'./data/X_y.pt{three_spd}'):
        dict_spd = read_matlab_files('/Users/baizhonghai/TP/RiemannianCovDs/data_RGB_canUse3elements/mat_ETH')
        # vectorize and connect each subset save as one vector
        if not three_spd:
            dict_spd = filter(dict_spd)
        dict_vector = vectorize_and_concatenate(dict_spd)
        X, y = split_to_X_y(dict_vector)

        X_y = {
            'X': X,
            'y': y,
        }
        torch.save(X_y, f'./data/X_y.pt{three_spd}')
    else:
        X_y = torch.load(f'./data/X_y.pt{three_spd}')
        X = X_y['X']
        y = X_y['y']
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)


        # Perform PCA
        '''pca = PCA(n_components=0.5)  # You can adjust the explained variance threshold as needed
        X_train_selected = pca.fit_transform(X_train_scaled)
        X_test_selected = pca.transform(X_test_scaled)
        '''
        #bestK
        selector_spd = SelectKBest(mutual_info_classif, k=top_k)
        X_train_selected = torch.tensor(selector_spd.fit_transform(X_train_scaled, y_train))
        X_test_selected = torch.tensor(selector_spd.transform(X_test_scaled))

        # Train SVM classifier
        #svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        #svm = SVC(kernel='linear', C=1.0, gamma='scale', random_state=42)
        svm = SVC(kernel='linear', C=2.0, gamma='scale', random_state=42)
        svm.fit(X_train_selected, y_train)

        # Make predictions
        y_pred = svm.predict(X_test_selected)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)


        # Example usage
        logger.info(f'paramers:three spd:{three_spd},topK:{top_k},accuracy:{accuracy}')
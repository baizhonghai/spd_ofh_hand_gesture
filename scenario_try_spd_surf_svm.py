from util.log_setup import logger
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from util.param_load import config


def check_record_correspond(spd, surf_bow):
    for i in range(len(spd['verify'])):
        if spd['verify'][i] != surf_bow['verify'][i] or spd['y'][i] != surf_bow['y'][i]:
            return False
    return True


def feature_combine(list_A, list_B):
    combined_list = [np.concatenate((a, b)) for a, b in zip(list_A, list_B)]
    return combined_list


if __name__ == '__main__':
    # todo 首先要确认所有记录是对齐的，要求在save的时候加上可比较的名字。
    spd = torch.load('./data/cambridge_1_spd_X_y_verify.pt')
    surf_bow = torch.load('./data/cambridge_surf_bow_X_y_verify.pt')
    if not check_record_correspond(spd, surf_bow):
        print('bad!!')
    # load surf_bow_features
    surf_X = surf_bow['X']
    # load spd_features
    spd_X = spd['X']
    Y = spd['y']
    # combine both kbest or not how many features each contributes.
    X = feature_combine(surf_X, spd_X)

    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=float(config['common_settings']['test_size']),
                                                        random_state=42)

    svm = SVC(C=10)
    svm.fit(X_train, Y_train)

    # Accuracy
    accuracy = svm.score(X_test, Y_test)
    logger.info(f'paramers:just combine without any tricky,accuracy:{accuracy}')

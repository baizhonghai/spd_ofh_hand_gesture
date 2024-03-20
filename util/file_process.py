import os
import numpy as np
import cv2


def read_files_to_list(directory_path):
    file_dict = {}
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        if os.path.isdir(filepath):
            label = filename
            for sub_filename in os.listdir(filepath):
                sub_filepath = os.path.join(filepath, sub_filename)
                if os.path.isdir(sub_filepath):
                    file_list = []
                    for finalfile in os.listdir(sub_filepath):
                        finalfile_path = os.path.join(sub_filepath, finalfile)
                        if os.path.isfile(finalfile_path):
                            file_list.append(finalfile_path)
                    if label in file_dict:
                        file_dict[label].append(file_list)
                    else:
                        file_dict[label] = []
                        file_dict[label].append(file_list)
    return file_dict


# 一个很不正宗的滑动窗口，长（或宽）除以长方向切割个数，余数是两连部分压着的区域
def calculate_slide_window(x, y, x_num, y_num):
    # 用一个matrix去存窗口的box格式就好（x,y,x+sizex,y+sizey）
    windows_matrix = [[[] for _ in range(x_num)] for _ in range(y_num)]

    size_x = x // x_num + x % x_num
    size_y = y // y_num + y % y_num
    x_start = 0
    y_start = 0

    for i in range(x_num):
        for j in range(y_num):
            windows_matrix[i][j] = [x_start, y_start, x_start + size_x - 1, y_start + size_y - 1]
            x_start = x_start + size_x - x % x_num
        y_start = y_start + size_y - y % y_num
        x_start = 0
    return windows_matrix

# todo, 需要对比看下原来matlab代码是不是这么写的，没实现resize，对比的时候一并看下增加上
def preprocess_dataset(dataset_dict, resize_x, resize_y, test=True):
    preprocessed_dataset_dict = {}
    for label, _ in dataset_dict.items():
        preprocessed_dataset_dict[label] = []
        for image_set in dataset_dict[label]:
            image_list = []
            for image_path in image_set:
                image = cv2.imread(image_path)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image_list.append(gray_image)
            preprocessed_dataset_dict[label].append(image_list)
        if test:
            return preprocessed_dataset_dict
    return preprocessed_dataset_dict


def read_matlab_files(directory_path):
    file_dict = {}
    for filename in sorted(os.listdir(directory_path)):
        filepath = os.path.join(directory_path, filename)
        if os.path.isdir(filepath):
            label = filename
            for sub_filename in sorted(os.listdir(filepath)):
                sub_filepath = os.path.join(filepath, sub_filename)
                if os.path.isdir(sub_filepath):
                    file_list = []
                    for finalfile in sorted(os.listdir(sub_filepath)):
                        finalfile_path = os.path.join(sub_filepath, finalfile)
                        if os.path.isfile(finalfile_path):
                            file_list.append(finalfile_path)
                    if label in file_dict:
                        file_dict[label].append(file_list)
                    else:
                        file_dict[label] = []
                        file_dict[label].append(file_list)
    return file_dict


def read_matlab_files_tmp(directory_path):
    file_dict = {}
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        if os.path.isdir(filepath):
            label = filename
            for sub_filename in os.listdir(filepath):
                sub_filepath = os.path.join(filepath, sub_filename)
                if os.path.isdir(sub_filepath):
                    file_list = []
                    for finalfile in os.listdir(sub_filepath):
                        finalfile_path = os.path.join(sub_filepath, finalfile)
                        if os.path.isfile(finalfile_path):
                            file_list.append(finalfile_path)
                    if label in file_dict:
                        file_dict[label].append(file_list)
                    else:
                        file_dict[label] = []
                        file_dict[label].append(file_list)
    return file_dict
if __name__ == '__main__':
    dataset_dict = read_files_to_list('../datasets/ETH-80')

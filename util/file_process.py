import os
import re

import numpy as np
import cv2


def read_videos_to_dict(directory_path, tmp=False):
    file_dict = {}
    for filename in sorted(os.listdir(directory_path)):
        filepath = os.path.join(directory_path, filename)
        if os.path.isdir(filepath):
            label = filename
            video_list = []
            for video_name in sorted(os.listdir(filepath)):
                video_list.append(os.path.join(directory_path, filepath, video_name))
            file_dict[label] = video_list
    if tmp:
        tmp_dict = {}
        tmp_dict['01'] = file_dict['01']
        tmp_dict['02'] = file_dict['02']
        return tmp_dict
    return file_dict


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






def build_target_directory(video_path, basic_path):
    parts = video_path.split('/')
    #basic_path = '/Users/xxxx/TP/HandGestureRecognition/datasets/Northwestern_Hand_Gesture_Gray_key_frames12/'
    directory = basic_path + parts[7] + '/' + parts[8].split('.')[0] + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

if __name__ == '__main__':
    dataset_dict = read_files_to_list('../datasets/ETH-80')

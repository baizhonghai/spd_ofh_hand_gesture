import numpy as np
import cv2
import os
import math
from util.feature_util import save_feature_to_path
from util.file_process import read_videos_to_dict


class OpticalFlow:
    key_frame_no = 12

    def __init__(self, key_frame_no, directory_path, feature_path):
        self.key_frame_no = key_frame_no
        self.directory_path = directory_path
        self.feature_path = feature_path

    def compute_optical_flow(self, prev_gray, next_gray):
        # Compute optical flow using Lucas-Kanade method
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Convert flow vectors to polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return mag, ang

    def compute_hoof(self, frames, num_frames=10, bins=8, region_size_x=80, region_size_y=80):
        # Initialize arrays to store optical flow vectors
        flows = []

        # Convert frames to grayscale
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        for i in range(1, num_frames):
            # Convert to grayscale
            gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            # Compute flow magnitudes and angles
            mag, ang = self.compute_optical_flow(prev_gray, gray)
            # Quantize angles into bins
            bins_range = np.linspace(0, 2 * np.pi, bins + 1)
            quantized_ang = np.digitize(ang, bins_range) - 1

            # Compute HOOF for each region
            hoof = np.zeros((int(frames[0].shape[0] / region_size_y), int(frames[0].shape[1] / region_size_x), bins))
            for y in range(0, frames[0].shape[0], region_size_y):
                for x in range(0, frames[0].shape[1], region_size_x):
                    ang_region = quantized_ang[y:y + region_size_y, x:x + region_size_x]
                    hist, _ = np.histogram(ang_region, bins=range(bins + 1))
                    hoof[int(y / region_size_y), int(x / region_size_x)] = hist / np.sum(hist)

            flows.append(hoof)

            # Update previous frame
            prev_gray = gray

        return np.array(flows)

    def frame_extract(self, video_path, target_frame_count=10):
        # 如果是文件夹那么就直接返回。
        if os.path.isdir(video_path):
            image_set = []
            for single_path in sorted(os.listdir(video_path)):
                image = cv2.imread(os.path.join(video_path, single_path))
                image_set.append(image)
            return image_set

        # Open the input video
        video_capture = cv2.VideoCapture(video_path)
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # Calculate frame sampling interval
        frame_ratio = frame_count / target_frame_count
        # Initialize variables
        sampled_frames = []
        # Read and sample frames
        for i in range(target_frame_count):
            frame_num = int(math.floor(frame_ratio * i))
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = video_capture.read()
            if not ret:
                break
            sampled_frames.append(frame)
        video_capture.release()
        return sampled_frames

    def feature_extract(self, file_dict):
        feature_dict = {}
        for label, paths in file_dict.items():
            if len(label) > 2:
                continue
            feature_vectors = []
            for video_path in paths:
                print(video_path)
                frames = self.frame_extract(video_path, self.key_frame_no)
                hoof_features = self.compute_hoof(frames, num_frames=len(frames))
                feature_vectors.append(hoof_features.flatten())
            feature_dict[label] = feature_vectors
        return feature_dict

    def split_to_X_y(self, dict_features):
        X = []
        y = []
        for label, class_set in dict_features.items():
            for feature_vector in class_set:
                X.append(feature_vector)
                y.append(int(label))
        return X, y

    def generate(self):
        file_dict = read_videos_to_dict(self.directory_path, tmp=False)
        feature_dict = self.feature_extract(file_dict)
        X, Y = self.split_to_X_y(feature_dict)
        save_feature_to_path(X, Y, Y, self.feature_path)

import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from util.file_process import build_target_directory
from util.file_process import read_videos_to_dict
import os
import time

# Step 1: Frame Extraction
def extract_frames(video_path, num_clusters):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    less_flag = False
    if total_frames < num_clusters:
        num_frames = num_clusters
        less_flag = True
    else:
        num_frames = total_frames
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.int32)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames, less_flag


def extract_frames_from_directory(frames_directory, num_clusters):
    frame_files = sorted(os.listdir(frames_directory))
    total_frames = len(frame_files)
    less_flag = False

    if total_frames < num_clusters:
        num_frames = num_clusters
        less_flag = True
    else:
        num_frames = total_frames

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.int32)
    frames = []

    for idx in frame_indices:
        frame_file = os.path.join(frames_directory, frame_files[idx])
        frame = cv2.imread(frame_file)
        if frame is not None:
            frames.append(frame)

    return frames, less_flag

def extract_frames_from_directory_tmp(frames_directory, num_clusters):
    frame_files = sorted(os.listdir(frames_directory))
    total_frames = len(frame_files)
    less_flag = True


    num_frames = total_frames

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.int32)
    frames = []

    for idx in frame_indices:
        frame_file = os.path.join(frames_directory, frame_files[idx])
        frame = cv2.imread(frame_file)
        if frame is not None:
            frames.append(frame)

    return frames, less_flag



# Step 2: Feature Extraction (e.g., using color histograms)
def extract_features(frames):
    features = []
    for frame in frames:
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.append(hist)
    return np.array(features)


# Step 3: K-Means Clustering
def cluster_frames(features, num_clusters, ):
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    kmeans.fit(features)
    return kmeans.labels_, kmeans.cluster_centers_


# Step 4: Selecting Key Frames
def select_key_frames(frames, labels, centroids, features):
    key_frames = []  # Initialize an empty list to store the selected key frames
    # frames = np.array(frames)
    key_frames_index = []
    # Iterate over each cluster centroid
    for label, centroid in enumerate(centroids):
        # Calculate distances from all frames in the cluster to the centroid
        cluster_features = features[labels == label]
        distances = cdist(cluster_features, [centroid])

        # Find the index of the frame closest to the centroid within the cluster
        closest_feature_idx = np.argmin(distances)

        # Retrieve the index of the frame within the original list of frames
        original_indices = np.where(labels == label)[0]
        closest_frame_idx = original_indices[closest_feature_idx]
        key_frames_index.append(closest_frame_idx)
    key_frames_index = sorted(key_frames_index)
    # Append the closest frame to the key_frames list
    key_frames = [frames[i] for i in key_frames_index]

    return key_frames


def whole_process_for_one_video(video_path, num_clusters, basic_path):
    start_time = time.perf_counter()
    frames, less_flag = extract_frames_from_directory(video_path,
                                                      num_clusters)  # extract_frames(video_path, num_clusters)
    '''
    for the time info just temp 
    start_time = time.perf_counter()
    frames, less_flag = extract_frames_from_directory_tmp(video_path,
                                                      num_clusters)
    '''
    if not less_flag:
        features = extract_features(frames)
        labels, centers = cluster_frames(features, num_clusters)
        key_frames = select_key_frames(frames, labels, centers, features)
    else:
        key_frames = frames
    end_time = time.perf_counter() #

    # Calculate the time cost
    time_cost = end_time - start_time#

    print(f"Time cost: {time_cost} seconds")#
    target_path = build_target_directory(video_path, basic_path=basic_path)
    for i, frame in enumerate(key_frames):
        cv2.imwrite(f'{target_path}frame_{i:02d}.jpg', frame)


if __name__ == '__main__':
    basic_path = '/Users/xxx/TP/HandGestureRecognition/datasets/Cambridge_Hand_Gesture_keyframe_kmeanscolor04/'
    num_clusters = 6
    directory_path = '/Users/xxx/TP/HandGestureRecognition/datasets/Cambridge_Hand_Gesture'



    file_dict = read_videos_to_dict(directory_path, tmp=False)
    for key, class_set in file_dict.items():
        for path in class_set:
            print(path)
            whole_process_for_one_video(path, num_clusters, basic_path)
    # Now you can use key_frames for classification or further processing

''' directory_path = '/Users/xxx/TP/HandGestureRecognition/datasets/Northwestern_Hand_Gesture'
    file_dict = read_videos_to_dict(directory_path, tmp=False)
    for key, class_set in file_dict.items():
        for path in class_set:
            select_optical_flow_key_frames(path)'''

import cv2
import numpy as np
from sklearn.cluster import KMeans
from util.file_process import read_videos_to_dict


class KeyFrameExtractorWithKMeans:
    def __init__(self, num_clusters=5):
        self.num_clusters = num_clusters

    def extract_optical_flow(self, video_path):
        cap = cv2.VideoCapture(video_path)
        all_flows = []
        prev_frame = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if prev_frame is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    None,  # Previous flow
                    0.5,  # Pyramids scale factor
                    3,  # Number of pyramid layers
                    15,  # Window size
                    3,  # Number of iterations
                    5,  # Size of the pixel neighborhood
                    1.2,  # Standard deviation for Gaussian filtering
                    0  # Flags
                )
                all_flows.append(flow)
            prev_frame = frame
        cap.release()
        return np.concatenate(all_flows, axis=0)

    def select_key_frames(self, video_paths, num_frames=10):
        all_flows = []
        for video_path in video_paths:
            all_flows.append(self.extract_optical_flow(video_path))

        # Perform k-means clustering on the optical flow vectors
        all_flows = np.concatenate(all_flows, axis=0)
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        kmeans.fit(all_flows.reshape(-1, 2))

        # Get cluster centroids
        cluster_centers = kmeans.cluster_centers_

        # Calculate distances of each flow vector to cluster centroids
        distances = kmeans.transform(all_flows.reshape(-1, 2))

        # Assign frames to the nearest cluster centroid
        frame_indices = np.argmin(distances, axis=1)

        # Choose key frames as the frames closest to each cluster centroid
        key_frames = []
        for i in range(self.num_clusters):
            cluster_frames = all_flows[frame_indices == i]
            representative_frame_index = np.argmax(np.linalg.norm(cluster_frames - cluster_centers[i], axis=1))
            key_frames.append(cluster_frames[representative_frame_index])

        return key_frames


# Example usage
if __name__ == "__main__":
    key_frame_extractor = KeyFrameExtractorWithKMeans(num_clusters=5)

    # video_paths = ['video1.mp4', 'video2.mp4', 'video3.mp4']  # List of paths to training videos
    directory_path = '/Users/xxx/TP/HandGestureRecognition/datasets/Northwestern_Hand_Gesture'
    file_dict = read_videos_to_dict(directory_path, tmp=False)
    for key, class_set in file_dict.items():
        for path in class_set:
            select_optical_flow_key_frames(path)



    num_frames = 10
    key_frames = key_frame_extractor.select_key_frames(video_paths, num_frames)

    # 我们将kmeans生成的关键帧重新写入到新的文件夹，这样就是跟下一个环节拆开了。

    # Now key_frames contains the selected key frames based on k-means clustering of optical flow across multiple training videos

    '''    print(len(key_frames))
    for i, frame in enumerate(key_frames):
        cv2.imwrite(f'{basic_path}frame_{i}.jpg', frame)'''
import cv2
import numpy as np


'''def compute_optical_flow(prev_gray, next_gray):
    # Compute optical flow using Lucas-Kanade method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Convert flow vectors to polar coordinates
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return mag, ang'''


class KmeansKeyFrame:
    def __init__(self, video_path, num_frames):
        self.num_frames = num_frames
        self.video_path = video_path

    # Function to extract optical flow features from two frames
    def extract_optical_flow(self, frame1, frame2):
        flow = cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY),
            None,  # Previous flow
            0.5,  # Pyramids scale factor
            3,  # Number of pyramid layers
            15,  # Window size
            3,  # Number of iterations
            5,  # Size of the pixel neighborhood
            1.2,  # Standard deviation for Gaussian filtering
            0  # Flags
        )
        return flow

    def compute_flow_magnitude(self, flow):
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        return magnitude

    def select_key_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_magnitudes = []
        prev_frame = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if prev_frame is not None:
                flow = self.extract_optical_flow(prev_frame, frame)
                magnitude = self.compute_flow_magnitude(flow)
                avg_magnitude = np.mean(magnitude)
                frame_magnitudes.append((avg_magnitude, frame))
            prev_frame = frame
        cap.release()

        # Sort frames based on average flow magnitude
        frame_magnitudes.sort(key=lambda x: x[0], reverse=True)

        # Select top N frames with highest motion magnitude
        key_frames = [frame for _, frame in frame_magnitudes[:self.num_frames]]

        return key_frames

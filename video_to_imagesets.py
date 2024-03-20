import os
import ffmpeg


def extract_frames(input_video, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ffmpeg.input(input_video).output(os.path.join(output_dir, 'frame_%04d.jpg')).run()


def transfer_video_to_image_sets(source_dir, output_dir_base):
    for sub_dir in sorted(os.listdir(source_dir)):
        for filename in sorted(os.listdir(os.path.join(source_dir, sub_dir))):
            if filename.endswith('.avi'):
                video_path = os.path.join(source_dir, sub_dir, filename)
                video_name = os.path.splitext(filename)[0]
                output_dir = os.path.join(output_dir_base, sub_dir, video_name)
                extract_frames(video_path, output_dir)


if __name__ == "__main__":
    '''directory = '/Users/baizhonghai/TP/HandGestureRecognition/datasets/Northwestern_Hand_Gesture/'
    out_directory_base = '/Users/baizhonghai/TP/HandGestureRecognition/datasets/Northwestern_Hand_Gesture_IMG'''

    directory = '/Users/baizhonghai/TP/HandGestureRecognition/datasets/HandGesture/'
    out_directory_base = '/Users/baizhonghai/TP/HandGestureRecognition/datasets/HandGesture_IMG/'

    transfer_video_to_image_sets(directory, out_directory_base)

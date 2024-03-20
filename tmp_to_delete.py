import cv2

def get_image_size(image_path):
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    return width, height

if __name__=='__main__':
    #image_path = "/Users/baizhonghai/TP/HandGestureRecognition/datasets/Northwestern_Hand_Gesture_IMG/01/01_Fist_01/frame_0001.jpg"  # Change this to the path of your image file
    image_path = "/Users/baizhonghai/TP/HandGestureRecognition/datasets/Cambridge_Hand_Gesture/1/Set1_1_0001/frame-0000.jpg"
    width, height = get_image_size(image_path)
    print("Width:", width)
    print("Height:", height)
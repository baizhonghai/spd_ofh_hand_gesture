import matplotlib.pyplot as plt
import os

if __name__=='__main__':
    # Path to the directory containing subdirectories
    base_path = "/Users/xxxx/TP/HandGestureRecognition/datasets/Northwestern_Hand_Gesture_IMG/"

    # Gesture types
    gestures = ['Fist', 'Hand', 'Hold', 'Index', 'SideHand', 'SideIndex', 'Thumb', 'Fist', 'Hand', 'Hold']

    # Create a figure and axes for subplots
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    # Flatten axes array to simplify indexing
    axes = axes.flatten()

    # Loop through each subfigure
    for i in range(1, 11):
        # Construct the path to the subdirectory for the current subfigure
        sub_dir_path = os.path.join(base_path, f"{i:02d}/{i:02d}_{gestures[i - 1]}_01/")

        # Check if the subdirectory exists
        if os.path.exists(sub_dir_path):
            # List files in the subdirectory
            files = sorted(os.listdir(sub_dir_path))

            # Choose the first image file (you may need to adjust this)
            if files:
                img_path = os.path.join(sub_dir_path, files[0])

                # Load and display the image on the corresponding subplot
                img = plt.imread(img_path)
                axes[i - 1].imshow(img)
                #axes[i - 1].set_title(f"category {i} - {gestures[i - 1]}")
                axes[i - 1].set_title(f"category {i}")
                # Remove axes
                axes[i-1].axis('off')
        else:
            # If the subdirectory doesn't exist, display an empty plot
            axes[i - 1].axis('off')
            axes[i - 1].set_title(f"category {i} - Not Available")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

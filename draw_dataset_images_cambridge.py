import os.path
import matplotlib.pyplot as plt
from PIL import Image

import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Base path for the images
    base_path = "/Users/xxx/TP/HandGestureRecognition/datasets/Cambridge_Hand_Gesture/{}/Set1_{}_0000/"

    # Create a figure with 3x6 subplots (to accommodate 18 images in three rows)
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))

    # Loop through each index to generate image paths and plot
    prev_subdir = None  # Store the previous subdirectory index
    row = 0
    for idx, ax in enumerate(axes.flat, start=1):
        # Calculate subdirectory index
        subdir_index = (idx + 1) // 2

        # Generate image paths
        image_path_base = base_path.format(subdir_index, subdir_index)

        # Load images
        dir_paths = sorted(os.listdir(image_path_base))
        frame_1 = dir_paths[0]
        frame_2 = dir_paths[-1]
        image_path_1 = os.path.join(image_path_base, frame_1)
        image_path_2 = os.path.join(image_path_base, frame_2)
        image_1 = Image.open(image_path_1)
        image_2 = Image.open(image_path_2)

        # Plot the first image
        if idx % 2 != 0:  # Plot on odd-numbered subplots
            ax.imshow(image_1)
            ax.set_title(f"category {subdir_index}, start state")
        # Plot the second image
        else:  # Plot on even-numbered subplots
            ax.imshow(image_2)
            ax.set_title(f"category {subdir_index}, end state")

        # Remove axes
        ax.axis('off')

        # Adjust the position of subplots within the same row
        if subdir_index == prev_subdir:
            ax.set_position(ax.get_position().translated(-0.0, 0))  # Move the subplot to the left

        # Update previous subdirectory index and row
        prev_subdir = subdir_index
        if idx % 6 == 0:
            row += 1

    # Adjust layout
    plt.tight_layout()
    plt.show()

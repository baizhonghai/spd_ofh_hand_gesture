from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')

# 代码写着写着就感觉像一坨屎了，无语
def print_confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    # Calculate accuracy for each label
    num_labels = len(cm)
    for i in range(num_labels):
        label_accuracy = cm[i, i] / sum(cm[i, :])
        print(f"Accuracy for label {i}: {label_accuracy:.2f}")


def show_confusion_matrix_colorful(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    # Generate consecutive labels based on the size of the confusion matrix
    labels = [str(i) for i in range(len(cm))]

    # Plot confusion matrix
    plt.figure(figsize=(8, 8))
    sns.set(font_scale=2.0)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=labels, yticklabels=labels,cbar=False)

    # Move x-axis tick labels to the top
    # plt.tick_params(axis='x', which='both', bottom=False, top=True)

    # Add title and labels
    plt.title('Confusion Matrix on ' + title + ' dataset')

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    #plt.gca().xaxis.set_ticks_position('top')
    plt.show()


def print_mean_std(accuracy_list):
    mean_value = np.mean(accuracy_list)
    std_deviation = np.std(accuracy_list)

    print("Mean:", mean_value)
    print("Standard Deviation:", std_deviation)

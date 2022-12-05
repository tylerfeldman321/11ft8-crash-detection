import pandas as pd
import numpy as np
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# SOUND_DATA_CSV = os.path.join()  # TODO path to csv with sound data
SIGN_DETECTION_RESULTS_CSV = os.path.join('data', 'sign_detection_results.csv')
BAR_SIM_RESULTS_CSV = os.path.join('data', 'ssim.csv')
LABELS_CSV = os.path.join('data', 'labels.csv')


def load_dataset(verbose=True, show_results=True):

    if verbose: print('Loading dataset...')

    # sound_data_df = pd.read_csv()  # TODO load sound data
    video_names = pd.read_csv(LABELS_CSV).columns.values[1:]

    sign_detection_arr = pd.read_csv(SIGN_DETECTION_RESULTS_CSV)
    ssim_arr = pd.read_csv(BAR_SIM_RESULTS_CSV)
    labels = pd.read_csv(LABELS_CSV)

    X, y = None, None

    for video_name in video_names:
        sign_detection_results = sign_detection_arr[video_name].values
        ssim_results = ssim_arr[video_name].values
        video_labels = labels[video_name].values

        variance = extract_variance_of_moving_window(sign_detection_results)

        if show_results:
            plot_features_for_video(video_name, ssim_results, sign_detection_results, variance, video_labels)

        video_data = np.vstack((variance, ssim_results))  # np.expand_dims(variance, axis=0)  # TODO: add in sound data

        if X is None or y is None:
            X, y = video_data, video_labels
        else:
            X, y = np.hstack((X, video_data)), np.hstack((y, video_labels))

    if verbose:
        print('--------------------------')
        print(f'Loaded dataset with data of shape = {X.shape}, and labels of shape: {y.shape}')
        print(f'Number of samples: {X.shape[0]}, number of features: {X.shape[1]}')
        print(f'Number of true samples: {int(np.sum(y))}')

    X, y = X.T, y.T
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if verbose:
        print(f'Train Dataset: {X_train.shape}, Train Labels: {y_train.shape}, Num True Samples: {np.sum(y_train)}')
        print(f'Test Dataset: {X_test.shape}, Test Labels: {y_test.shape}, Num True Samples: {np.sum(y_test)}')
        print('--------------------------')

    return X_train, X_test, y_train, y_test


def plot_features_for_video(video_name, ssim_results, sign_detection_results, variance, video_labels):
    plt.figure()
    plt.title(video_name)
    plt.plot(np.arange(0, len(ssim_results)), ssim_results, 'k-', label='Bar Similarity')
    plt.plot(np.arange(0, len(sign_detection_results)), sign_detection_results, 'b-', label='Template Matching')
    plt.plot(np.arange(0, len(variance)), variance, 'r-', label='Variance')
    plt.plot(np.arange(0, len(video_labels)), video_labels, 'g-', label='Label')
    plt.legend(loc='best')
    plt.show()


def plot_dataset(dataset):
    X_train, X_test, y_train, y_test = dataset

    X = np.vstack((X_train, X_test))
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # just plot the dataset first
    plt.figure()
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#0000FF", "#0000FF"])
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], c=y_train[y_train == 1], cmap=cm_bright, edgecolors="k")
    # Plot the testing points
    # plt.scatter(
    #     X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], c=y_test[y_test == 1], cmap=cm_bright, alpha=0.6, edgecolors="k"
    # )
    plt.title('Dataset')
    plt.xlabel('Sign Detection Variance')
    plt.ylabel('Bar Similarity')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()


def extract_variance_of_moving_window(sign_detection_results, window_size=500):
    variance = np.zeros(sign_detection_results.shape)
    for i, sign_detection_val in enumerate(sign_detection_results):
        if i - window_size < 0:
            var = np.var(sign_detection_results[0:i+1])
        else:
            var = np.var(sign_detection_results[i-window_size:i+1])
        variance[i] = var
    return variance


if __name__ == '__main__':
    dataset = load_dataset()
    plot_dataset(dataset)

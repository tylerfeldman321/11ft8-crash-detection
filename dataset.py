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


def load_dataset(verbose=True):

    if verbose: print('Loading dataset...')

    # sound_data_df = pd.read_csv()  # TODO load sound data
    sign_detection_arr = pd.read_csv(SIGN_DETECTION_RESULTS_CSV).to_numpy()
    ssim_arr = pd.read_csv(BAR_SIM_RESULTS_CSV).to_numpy()
    labels = pd.read_csv(LABELS_CSV).to_numpy()

    X, y = None, None

    for col_idx in range(1, labels.shape[1]):
        sign_detection_results = sign_detection_arr[:, col_idx]
        ssim_results = ssim_arr[:, col_idx]

        sign_detection_results = extract_variance_of_moving_window(sign_detection_results)

        video_data = np.vstack((sign_detection_results, ssim_results))  # TODO: add in sound data
        video_labels = labels[:, col_idx]

        if X is None or y is None:
            X, y = video_data, video_labels
        else:
            X, y = np.hstack((X, video_data)), np.hstack((y, video_labels))

    X, y = X.T, y.T
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if verbose:
        print('--------------------------')
        print(f'Loaded dataset with data of shape = {X.shape}, and labels of shape: {y.shape}')
        print(f'Number of samples: {X.shape[0]}, number of features: {X.shape[1]}')
        print(f'Number of true samples: {int(np.sum(y))}')
        print(f'Train Dataset: {X_train.shape}, Train Labels: {y_train.shape}, Num True Samples: {np.sum(y_train)}')
        print(f'Test Dataset: {X_test.shape}, Test Labels: {y_test.shape}, Num True Samples: {np.sum(y_test)}')
        print('--------------------------')

    return X_train, X_test, y_train, y_test


def get_negative_and_positive_samples(dataset):
    X_train, X_test, y_train, y_test = dataset
    X_train_0, y_train_0 = X_train[y_train == 0], y_train[y_train == 0]
    X_train_1, y_train_1 = X_train[y_train == 1], y_train[y_train == 1]
    X_test_0, y_test_0 = X_test[y_test == 0], y_test[y_test == 0]
    X_test_1, y_test_1 = X_test[y_test == 1], y_test[y_test == 1]
    return X_train_0, y_train_0, X_train_1, y_train_1, X_test_0, y_test_0, X_test_1, y_test_1



def plot_dataset(dataset):
    X_train, X_test, y_train, y_test = dataset

    X = np.vstack((X_train, X_test))
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # just plot the dataset first
    plt.figure()
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    # Plot the testing points
    plt.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
    )
    plt.xlabel('Sign Detection')
    plt.ylabel('Bar Similarity')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def extract_variance_of_moving_window(sign_detection_results, window_size=500, show=False):
    variance = np.zeros(sign_detection_results.shape)
    for i, sign_detection_val in enumerate(sign_detection_results):
        if i - window_size < 0:
            var = np.var(sign_detection_results[0:i+1])
        else:
            var = np.var(sign_detection_results[i-window_size:i])
        variance[i] = var

    if show:
        plt.figure()
        plt.plot(np.arange(0, len(sign_detection_results)), sign_detection_results)

        plt.figure()
        plt.plot(np.arange(0, len(variance)), variance)
        plt.show()

    return variance


if __name__ == '__main__':
    dataset = load_dataset()
    plot_dataset(dataset)

import pandas as pd
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from generator import TIMESTAMPS_CSV


SOUND_DATA_CSV = os.path.join('data', 'audio.csv')
SIGN_DETECTION_RESULTS_CSV = os.path.join('data', 'sign_detection_variance.csv')
BAR_SIM_RESULTS_CSV = os.path.join('data', 'ssim.csv')
LABELS_CSV = os.path.join('data', 'labels.csv')
TRAIN_PERCENTAGE = 0.75


def load_dataset(verbose=False, show_results=True, seed=0):
    """ Loads the dataset and labels into a train and test split """

    if verbose:
        print('Loading dataset...')

    video_names = pd.read_csv(LABELS_CSV).columns.values[1:]
    np.random.seed(seed)
    np.random.shuffle(video_names)

    sound_df = pd.read_csv(SOUND_DATA_CSV)
    sign_detection_variance_df = pd.read_csv(SIGN_DETECTION_RESULTS_CSV)
    ssim_df = pd.read_csv(BAR_SIM_RESULTS_CSV)
    labels_df = pd.read_csv(LABELS_CSV)

    X_train, y_train, X_test, y_test = None, None, None, None
    num_train_videos = len(video_names) * TRAIN_PERCENTAGE
    test_data_lengths = []
    test_data_names = []

    for video_name in video_names:
        variance = sign_detection_variance_df[video_name].values
        ssim_results = ssim_df[video_name].values
        video_labels = labels_df[video_name].values
        sound = sound_df[video_name].values

        if show_results:
            plot_features_for_video(video_name=video_name, ssim_results=ssim_results, sign_detection_results=None, variance=variance, sound_data=sound, video_labels=video_labels)

        video_data = np.vstack((variance, ssim_results, sound))
        if num_train_videos > 0:
            if X_train is None or y_train is None:
                X_train, y_train = video_data, video_labels
            else:
                X_train, y_train = np.hstack((X_train, video_data)), np.hstack((y_train, video_labels))
            num_train_videos -= 1
        else:
            if X_test is None or y_test is None:
                X_test, y_test = video_data, video_labels
            else:
                X_test, y_test = np.hstack((X_test, video_data)), np.hstack((y_test, video_labels))
            test_data_names.append(video_name)
            test_data_lengths.append(video_labels.size)

    X_train, y_train, X_test, y_test = X_train.T, y_train.T, X_test.T, y_test.T

    if verbose:
        print(f'Train Dataset: {X_train.shape}, Train Labels: {y_train.shape}, Num True Samples: {np.sum(y_train)}')
        print(f'Test Dataset: {X_test.shape}, Test Labels: {y_test.shape}, Num True Samples: {np.sum(y_test)}')
        print('--------------------------')

    return X_train, X_test, y_train, y_test, test_data_lengths, test_data_names


def plot_features_for_video(video_name, ssim_results=None, sign_detection_results=None, variance=None, sound_data=None, video_labels=None):
    """ Plot extracted features over time """
    plt.figure()
    plt.title(f'Features vs. Frame Number for {video_name}')
    if ssim_results is not None: 
        plt.plot(np.arange(0, len(ssim_results)), ssim_results, 'k-', label='Bar Similarity')
    if sign_detection_results is not None:
        plt.plot(np.arange(0, len(sign_detection_results)),
                sign_detection_results, 'b-', label='Template Matching')
    if variance is not None:
        plt.plot(np.arange(0, len(variance)), variance, 'r-', label='Variance')
    if sound_data is not None:
        plt.plot(np.arange(0, len(sound_data)), sound_data, 'c-', label='Sound')
    if video_labels is not None:
        plt.plot(np.arange(0, len(video_labels)), video_labels, 'g-', label='Label')
    plt.xlabel('Frame')
    plt.ylabel('Feature Value')
    plt.legend(loc='best')
    plt.show()


def load_timestamps():
    """ Loads the frame timestamp labels as a dictionary indexed by video name """
    timestamps_df = pd.read_csv(TIMESTAMPS_CSV)
    timestamps_dict = {}
    for file, frame in zip(timestamps_df['file'], timestamps_df['frame']):
        timestamps_dict[file] = frame
    return timestamps_dict


if __name__ == '__main__':
    dataset = load_dataset()

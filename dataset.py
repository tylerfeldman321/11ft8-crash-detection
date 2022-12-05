import pandas as pd
import numpy as np
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split


# SOUND_DATA_CSV = os.path.join()  # TODO path to csv with sound data
SIGN_DETECTION_RESULTS_CSV = os.path.join('data', 'sign_detection_results.csv')
BAR_SIM_RESULTS_CSV = os.path.join('data', 'ssim.csv')
LABELS_CSV = os.path.join('data', 'labels.csv')


def load_dataset(verbose=True):

    if verbose: print('Loading dataset...')

    # sound_data_df = pd.read_csv()  # TODO load sound data
    sign_detection = pd.read_csv(SIGN_DETECTION_RESULTS_CSV).to_numpy()
    ssim = pd.read_csv(BAR_SIM_RESULTS_CSV).to_numpy()
    labels = pd.read_csv(LABELS_CSV).to_numpy()

    X, y = None, None

    for col_idx in range(1, labels.shape[1]):

        video_data = np.vstack((sign_detection[:, col_idx],
                              ssim[:, col_idx]))  # TODO: add in sound data
        video_labels = labels[:, col_idx]

        # TODO Do processing on data here?

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


if __name__ == '__main__':
    load_dataset()

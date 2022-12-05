import pandas as pd
import numpy as np
import os
import glob
import pandas as pd


# SOUND_DATA_CSV = os.path.join()  # TODO path to csv with sound data
SIGN_DETECTION_RESULTS_CSV = os.path.join('data', 'sign_detection_results.csv')
BAR_SIM_RESULTS_CSV = os.path.join('data', 'ssim.csv')
LABELS_CSV = os.path.join('data', 'labels.csv')


def load_dataset():

    # sound_data_df = pd.read_csv()  # TODO load sound data
    sign_detection = pd.read_csv(SIGN_DETECTION_RESULTS_CSV).to_numpy()
    ssim = pd.read_csv(BAR_SIM_RESULTS_CSV).to_numpy()
    labels = pd.read_csv(LABELS_CSV).to_numpy()

    X, Y = None, None

    for col_idx in range(labels.shape[1]):
        video_data = np.vstack((sign_detection[:, col_idx],
                              ssim[:, col_idx]))  # TODO: add in sound data
        video_labels = labels[:, col_idx]

        # TODO Do processing on data here?

        if X is None or Y is None:
            X, Y = video_data, video_labels
        else:
            X, Y = np.hstack((X, video_data)), np.hstack((Y, video_labels))

    print(f'Loaded dataset with data of shape = {X.shape}, and labels of shape: {Y.shape}')

    return X, Y


if __name__ == '__main__':
    load_dataset()

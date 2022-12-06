import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from dataset import load_dataset, plot_features_for_video
import time
import pickle
import os
from audio_processing.video_to_wav import get_normalized_audio_amplitude
from crash_bar_processing.crash_bar_ssim import CrashBarSSIM
from sign_detection.sign_detector import SignDetector
import argparse


class CrashPredictor:

    THRESHOLD = 100

    def __init__(self, clf=None):
        if clf:
            self.clf = clf
        else:
            # self.clf = SVC(gamma=2, C=1, verbose=True)
            self.clf = MLPClassifier(max_iter=500, learning_rate='adaptive')

    def train(self, dataset, verbose=True, train_all=False):
        """ Train the model on a set of extracted features """
        if verbose:
            print('Training model...')

        X_train, X_test, y_train, y_test, _, _ = dataset

        if train_all:
            X_all = np.vstack((X_train, X_test))
            y_all = np.append(y_train, y_test)
            self.clf.fit(X_all, y_all)
        else:
            self.clf.fit(X_train, y_train)

    def save_model(self, save_path='mlp.pkl'):
        pickle.dump(self.clf, open(save_path, 'wb'))

    def load_model(self, model_path='mlp.pkl'):
        self.clf = pickle.load(open(model_path, 'rb'))

    def test(self, dataset, verbose=True):
        """ Evaluate the model by running on unseen testing set """

        if verbose:
            print('Testing model...')

        X_train, X_test, y_train, y_test, test_data_lengths, test_data_names = dataset
        start = 0
        score_negative_samples = []
        score_positive_samples = []
        len_neg_samples = 0
        len_pos_samples = 0
        # predictions = self.clf.predict_proba(X_test)[:, 1]
        for length, name in zip(test_data_lengths, test_data_names):
            X, y = self.split_data_by_video(X_test, y_test, start, length)
            start += length

            predictions = self.clf.predict(X)
            score = self.clf.score(X, y)

            negative_samples, negative_labels = X[y == 0], y[y == 0]
            score_negative_samples.append(self.clf.score(negative_samples, negative_labels))
            len_neg_samples += len(negative_samples)

            positive_samples, positive_labels = X[y == 1], y[y == 1]
            score_positive_samples.append(self.clf.score(positive_samples, positive_labels))
            len_pos_samples += len(positive_samples)

            self.plot_predictions(predictions, y, name)
            timestamps = self.generate_timestamps(predictions)
            times = self.convert_to_hour_minute_second(timestamps)
            print(f'Predicted crashes at {times} for video {name}.')

        if verbose:
            print(f'Overall Score: {score}')
            print(f'Score on Negative Samples: {np.mean(score_negative_samples)}')
            print(f'Number of Negative Samples: {len_neg_samples}')
            print(f'Score on Positive Samples: {np.mean(score_positive_samples)}')
            print(f'Number of Postive Samples: {len_pos_samples}')

        return predictions

    def split_data_by_video(self, X, y, start, length):
        """ Split the data by video """
        split_x = X[start:start + length]
        split_y = y[start:start + length]
        return split_x, split_y

    def generate_timestamps(self, predictions):
        """ Generate timestamps from predictions. Filter out predictions that are close to one another """
        frames = []
        for i, pred in enumerate(predictions):
            if pred == 1.0:  # Predicted crash
                frames.append(i)
        copy = []
        for frame in frames:
            flag = True
            for element in copy:
                if np.abs(element - frame) < self.THRESHOLD:
                    flag = False
            if flag:
                copy.append(frame)

        timestamps = [self.convert_frame_to_timestamp(frame) for frame in copy]
        return timestamps

    def convert_frame_to_timestamp(self, frame_number, fps=15):
        """ Convert frame number to timestamp in seconds """
        return frame_number/fps

    def convert_to_hour_minute_second(self, timestamps):
        """ Convert List of timestamps (in seconds) to hour:minute:second format """
        times = [time.strftime('%H:%M:%S', time.gmtime(timestamp)) for timestamp in timestamps]
        return times

    def plot_predictions(self, predictions, labels, name):
        """ Plot model predictions and labels """
        plt.figure()
        plt.title(f'Prediction results for {name}')
        plt.plot(np.arange(0, len(predictions)), predictions, label='predictions')
        plt.plot(np.arange(0, len(labels)), labels, label='labels', zorder=10)
        plt.legend()
        plt.xlabel('Frame')
        plt.ylabel('Crash likelihood')
        plt.show()

    def __call__(self, *args):
        video_file_path = args[0]
        if not os.path.exists(video_file_path):
            raise Exception(f'Provided video does not exist: {video_file_path}')
        if not video_file_path.endswith('mp4'):
            raise Exception('Provided file is not in mp4 format.')

        data = self.extract_features(video_file_path)

        predictions = self.clf.predict(data)
        timestamps = self.generate_timestamps(predictions)
        times = self.convert_to_hour_minute_second(timestamps)
        print('Predicted Crashes: ', end='')
        print(*times)

    def extract_features(self, video_file_path, show=False):
        # TODO: multiprocessing?
        audio_data = get_normalized_audio_amplitude(video_file_path, os.path.splitext(os.path.basename(video_file_path))[0])
        template_matching_variance = SignDetector().process_video(video_file_path, compute_variance=True)
        ssim, fps = CrashBarSSIM().detect(video_file_path)
        data = np.stack((template_matching_variance, ssim, audio_data), axis=1)
        if show:
            plot_features_for_video(video_file_path, ssim_results=ssim, variance=template_matching_variance, sound_data=audio_data)
        return data


def train_all_and_save_model():
    cp = CrashPredictor()
    dataset = load_dataset(show_results=False)
    cp.train(dataset, train_all=True)
    cp.save_model()


def train_and_test():
    cp = CrashPredictor()
    dataset = load_dataset(show_results=False)
    cp.train(dataset)
    pred = cp.test(dataset, verbose=True)


def run_cross_validation():
    # TODO: do this and get results
    return


def main():
    parser = argparse.ArgumentParser(description='Detect pixel changes in a video file')
    parser.add_argument('file', help='Path to video file')
    args = parser.parse_args()
    cp = CrashPredictor()
    cp.load_model()
    cp(args.file)


if __name__ == '__main__':
    main()

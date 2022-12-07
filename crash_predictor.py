import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from dataset import load_dataset, plot_features_for_video, load_timestamps
import time
import pickle
import os
from audio_processing.audio_processing import get_normalized_audio_amplitude
from crash_bar_processing.crash_bar_ssim import CrashBarSSIM
from sign_detection.sign_detector import SignDetector
import argparse
import cv2


class CrashPredictor:

    THRESHOLD = 900
    PRED_CORRECTNESS_THRESHOLD = 50  # In num of frames

    def __init__(self, clf=None):
        if clf:
            self.clf = clf
        else:
            self.clf = MLPClassifier(max_iter=500, learning_rate='adaptive')

    def train(self, dataset, verbose=False, train_all=False):
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

    def test(self, dataset, probability_threshold=0.5, show_results=False, verbose=False):
        """ Evaluate the model by running on unseen testing set """

        if verbose:
            print('Testing model...')

        X_train, X_test, y_train, y_test, test_data_lengths, test_data_names = dataset
        start = 0
        score_negative_samples = []
        score_positive_samples = []
        len_neg_samples = 0
        len_pos_samples = 0

        frame_prediction_dict = {}

        for length, name in zip(test_data_lengths, test_data_names):
            X, y = self.split_data_by_video(X_test, y_test, start, length)
            start += length

            probabilities = self.clf.predict_proba(X)[:, 1]  # self.clf.predict(X)
            predictions = (probabilities >= probability_threshold) * 1

            score = self.clf.score(X, y)

            negative_samples, negative_labels = X[y == 0], y[y == 0]
            score_negative_samples.append(self.clf.score(negative_samples, negative_labels))
            len_neg_samples += len(negative_samples)

            positive_samples, positive_labels = X[y == 1], y[y == 1]
            score_positive_samples.append(self.clf.score(positive_samples, positive_labels))
            len_pos_samples += len(positive_samples)

            if show_results:
                self.plot_predictions(predictions, y, name)

            frames_predictions = self.filter_raw_predictions(predictions, probabilities)
            timestamps = [self.convert_frame_to_timestamp(frame) for frame in frames_predictions]
            times = self.convert_to_hour_minute_second(timestamps)

            if verbose:
                print(f'Predicted crashes at {times} for video {name}.')

            frame_prediction_dict[name] = frames_predictions

        if verbose:
            print(f'Overall Score: {score}')
            print(f'Score on Negative Samples: {np.mean(score_negative_samples)}')
            print(f'Number of Negative Samples: {len_neg_samples}')
            print(f'Score on Positive Samples: {np.mean(score_positive_samples)}')
            print(f'Number of Postive Samples: {len_pos_samples}')

        return frame_prediction_dict

    def split_data_by_video(self, X, y, start, length):
        """ Split the data by video """
        split_x = X[start:start + length]
        split_y = y[start:start + length]
        return split_x, split_y

    def filter_raw_predictions(self, predictions, probabilities):
        """ Generate timestamps from predictions. Filter out predictions that are close to one another """        
        frame_predictions = np.where(predictions == 1.0)[0]
        frame_prediction_probabilities = probabilities[frame_predictions]

        if not len(frame_predictions):
            return []

        frame_prediction_probabilities_sorted, frame_predictions_sorted = zip(*sorted(zip(frame_prediction_probabilities, frame_predictions), reverse=True))

        frame_predictions_filtered = []
        frame_predictions_ignored = []
        for i in range(len(frame_predictions_sorted)):

            frame = frame_predictions_sorted[i]

            if frame in frame_predictions_ignored:
                continue

            frame_predictions_filtered.append(frame)

            for j in range(len(frame_predictions_sorted)):
                other_frame = frame_predictions_sorted[j]
                if np.abs(other_frame - frame) < self.THRESHOLD:
                    frame_predictions_ignored.append(other_frame)

        return frame_predictions_filtered

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
        """ Run inference on a video file """
        video_file_path = args[0]
        if not os.path.exists(video_file_path):
            raise Exception(f'Provided video does not exist: {video_file_path}')
        if not video_file_path.endswith('mp4'):
            raise Exception('Provided file is not in mp4 format.')

        capture = cv2.VideoCapture(video_file_path)
        fps = int(capture.get(cv2.CAP_PROP_FPS))

        data = self.extract_features(video_file_path)

        predictions = self.clf.predict(data)
        frames_predictions = self.filter_raw_predictions(predictions)
        timestamps = [self.convert_frame_to_timestamp(frame, fps) for frame in frames_predictions]
        times = self.convert_to_hour_minute_second(timestamps)
        print('Predicted Crashes: ', end='')
        print(*times)

    def extract_features(self, video_file_path, show=False):
        """ Extract features from a video file """
        # TODO: multiprocessing?
        audio_data = get_normalized_audio_amplitude(
            video_file_path, os.path.splitext(os.path.basename(video_file_path))[0])
        template_matching_variance = SignDetector().process_video(video_file_path,
                                                                  compute_variance=True)
        ssim, fps = CrashBarSSIM().detect(video_file_path)
        data = np.stack((template_matching_variance, ssim, audio_data), axis=1)
        if show:
            plot_features_for_video(video_file_path, ssim_results=ssim,
                                    variance=template_matching_variance, sound_data=audio_data)
        return data

    def evaluate_performance(self, dataset, probability_threshold=0.5, verbose=False):
        """ Evaluate the performance of the model

        Args:
            dataset (_type_): _description_
            probability_threshold (float, optional): Threshold above which to declare . Defaults to 0.5.
            verbose (bool, optional): Whether to print out detailed results. Defaults to False.

        Returns:
            Tuple: precision, recall
        """
        pred_crash_frames_dict = self.test(
            dataset, probability_threshold=probability_threshold, verbose=verbose)
        _, _, _, _, _, test_data_names = dataset
        true_crash_frames_dict = load_timestamps()

        tp, fp, fn = 0, 0, 0

        # Calculate precision and recall
        for video_name in test_data_names:
            pred_crash_frames = pred_crash_frames_dict[video_name]
            true_crash_frame = true_crash_frames_dict[video_name]  # Assuming only 1 true crash

            # If timestamp is not near the label, it is a false positive
            # If timestamp is near label, true positive
            # If no timestamp near label, false negative
            fp += len(pred_crash_frames)
            if len(pred_crash_frames) and self.found_true_crash(pred_crash_frames, true_crash_frame):
                tp += 1
                fp -= 1
            else:
                fn += 1

        precision, recall = compute_precision_and_recall(tp, fp, fn)

        if verbose:
            print(f'Precision: {precision}, Recall: {recall}')

        return precision, recall

    def found_true_crash(self, pred_crash_frames, true_crash_frame):
        """ Finds if we've found the true crash in the video

        Args:
            pred_crash_frames (List): _description_
            true_crash_frame (int): Frame on which the crash occurs

        Returns:
            bool: True if a prediction is within a threshold from the true crash frame
        """
        return np.abs(find_nearest(pred_crash_frames, true_crash_frame) - true_crash_frame) < self.PRED_CORRECTNESS_THRESHOLD


def compute_precision_and_recall(tp, fp, fn):
    """ Compute precision and recall

    Args:
        tp (int): Number of true positives
        fp (int): Number of false positives
        fn (int): Number of false negatives

    Returns:
        Tuple: (precision, recall)
    """
    if (tp + fp) != 0:
        precision = tp / (tp + fp)
    else:
        precision = 1.0
    recall = tp / (tp + fn)
    return precision, recall


def find_nearest(array, value):
    """ Finds nearest element in array to the value

    Args:
        array (ndarray): Numpy array
        value (float): Target value

    Returns:
        float: Element in array that is closest to value
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def main():
    parser = argparse.ArgumentParser(description='Detect pixel changes in a video file')
    parser.add_argument('file', help='Path to video file')
    args = parser.parse_args()
    cp = CrashPredictor()
    cp.load_model()
    cp(args.file)


if __name__ == '__main__':
    main()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from dataset import load_dataset
from sklearn.neighbors import KernelDensity
import seaborn as sns


class CrashPredictor:

    THRESHOLD = 100

    def __init__(self, clf=None):
        if clf:
            self.clf = clf
        else:
            # self.clf = SVC(gamma=2, C=1, verbose=True)
            self.clf = MLPClassifier(max_iter=500, learning_rate='adaptive')

    def train(self, dataset, verbose=True):
        if verbose:
            print('Training model...')

        X_train, X_test, y_train, y_test, _, _ = dataset
        self.clf.fit(X_train, y_train)

    def test(self, dataset, verbose=True):
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

            # cp.plot_predictions(predictions, y, name)
            timestamps = cp.generate_timestamps(predictions)
            print(f'Predicted crashes at {timestamps} for video {name}.')

        if verbose:
            print(f'Overall Score: {score}')
            print(f'Score on Negative Samples: {np.mean(score_negative_samples)}')
            print(f'Number of Negative Samples: {len_neg_samples}')
            print(f'Score on Positive Samples: {np.mean(score_positive_samples)}')
            print(f'Number of Postive Samples: {len_pos_samples}')
        return predictions

    def split_data_by_video(self, X, y, start, length):
        split_x = X[start:start + length]
        split_y = y[start:start + length]
        return split_x, split_y

    def generate_timestamps(self, predictions):
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
        return frame_number/fps

    def plot_predictions(self, predictions, labels, name):
        plt.figure()
        plt.title(f'Prediction results for {name}')
        plt.plot(np.arange(0, len(predictions)), predictions, label='predictions')
        plt.plot(np.arange(0, len(labels)), labels, label='labels', zorder=10)
        plt.legend()
        plt.xlabel('Frame')
        plt.ylabel('Crash likelihood')
        plt.show()


if __name__ == '__main__':
    cp = CrashPredictor()
    dataset = load_dataset(show_results=False)
    cp.train(dataset)
    pred = cp.test(dataset, verbose=True)

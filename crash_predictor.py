import numpy as np
from sklearn.svm import SVC
from dataset import load_dataset


class CrashPredictor:
    def __init__(self):
        self.clf = SVC(gamma=2, C=1)

    def train(self, dataset, verbose=True):
        if verbose: print('Training model...')

        X, y, X_train, X_test, y_train, y_test = dataset
        self.clf.fit(X_train, y_train)

    def test(self, dataset, verbose=True):
        X, y, X_train, X_test, y_train, y_test = dataset
        score = self.clf.score(X_test, y_test)

        negative_samples, negative_labels = X_test[y_test == 0], y_test[y_test == 0]
        score_negative_samples = self.clf.score(negative_samples, negative_labels)

        positive_samples, positive_labels = X_test[y_test == 1], y_test[y_test == 1]
        score_positive_samples = self.clf.score(positive_samples, positive_labels)

        if verbose:
            print(f'Overall Score: {score}')
            print(f'Score on Negative Samples: {score_negative_samples}')
            print(f'Number of Negative Samples: {len(negative_samples)}')
            print(f'Score on Positive Samples: {score_positive_samples}')
            print(f'Number of Postive Samples: {len(positive_samples)}')


if __name__ == '__main__':
    cp = CrashPredictor()
    dataset = load_dataset()
    cp.train(dataset)
    cp.test(dataset, verbose=True)

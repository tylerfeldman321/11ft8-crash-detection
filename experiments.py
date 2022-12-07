from crash_predictor import CrashPredictor
from dataset import load_dataset
import numpy as np
import matplotlib.pyplot as plt


def run_cross_validation(probability_threshold=0.5, num_folds=50):
    """ Perform random fold cross validation

    Args:
        num_folds (int, optional): _description_. Defaults to 5.
    """

    avg_precision = 0
    avg_recall = 0

    for i in range(num_folds):
        dataset = load_dataset(verbose=False, show_results=False, seed=i)
        crash_predictor = CrashPredictor()
        crash_predictor.train(dataset)
        p, r = crash_predictor.evaluate_performance(dataset, probability_threshold=probability_threshold)

        avg_precision += p / num_folds
        avg_recall += r / num_folds

    print(f'Average Precision: {avg_precision}, Average Recall: {avg_recall}')
    return avg_precision, avg_recall


def plot_precision_recall_for_varying_thresolds():
    probability_thresholds = np.linspace(0, 1.0, 51)
    
    precisions = np.zeros(len(probability_thresholds))
    recalls = np.zeros(len(probability_thresholds))
    f1_scores = np.zeros(len(probability_thresholds))

    for i, probability_threshold in enumerate(probability_thresholds):
        p, r = run_cross_validation(probability_threshold, num_folds=10)
        precisions[i] = p
        recalls[i] = r
        f1_scores[i] = 2 * (p * r) / (p + r)

    plt.plot(probability_thresholds, precisions, label='Average Precision')
    plt.plot(probability_thresholds, recalls, label='Average Recall')
    plt.plot(probability_thresholds, f1_scores, label='F1 Score')
    plt.title('Precision/Recall/F1 vs. Decision Threshold')
    plt.xlabel('Probability Decision Threshold')
    plt.ylabel('Precision/Recall/F1')
    plt.legend(loc='best')
    plt.show()

    np.save('precision.npy', precisions)
    np.save('recall.npy', recalls)
    np.save('f1.npy', f1_scores)


def train_all_and_save_model():
    cp = CrashPredictor()
    dataset = load_dataset(show_results=False)
    cp.train(dataset, train_all=True)
    cp.save_model()


def train_and_test():
    cp = CrashPredictor()
    dataset = load_dataset(show_results=False)
    cp.train(dataset)
    cp.test(dataset, verbose=True)


if __name__ == '__main__':
    plot_precision_recall_for_varying_thresolds()

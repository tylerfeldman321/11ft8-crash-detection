from crash_predictor import CrashPredictor
from dataset import load_dataset


def run_cross_validation(num_folds=20):
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
        p, r = crash_predictor.evaluate_performance(dataset)

        avg_precision += p / num_folds
        avg_recall += r / num_folds

    print(f'Average Precision: {avg_precision}, Average Recall: {avg_recall}')


if __name__ == '__main__':
    run_cross_validation()

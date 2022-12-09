from crash_predictor import CrashPredictor
from dataset import load_dataset


def train_all_and_save_model():
    cp = CrashPredictor()
    dataset = load_dataset(show_results=False)
    cp.train(dataset, train_all=True)
    cp.save_model()


if __name__ == '__main__':
    train_all_and_save_model()

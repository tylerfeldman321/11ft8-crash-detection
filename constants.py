# Paths to directories
DATA_DIR = 'data'
MODELS_DIR = 'models'

# Paths to csv files
CRASH_FOLDER = 'data/crash_samples/'
SSIM_CSV = 'data/ssim.csv'
TIMESTAMPS_CSV = 'data/timestamps.csv'
LABELS_CSV = 'data/labels.csv'
AUDIO_CSV = 'data/audio.csv'
SIGN_DETECTION_VARIANCE_CSV = 'data/sign_detection_variance.csv'

# Default probability threshold. Can be overriden with the -p argument on the command line
DEFAULT_PROBABILITY_THRESHOLD = 0.5

# Training set percentage from 0-1. The rest of the data will be used for the testing set
TRAIN_PERCENTAGE = 0.75

# Window to label crash frames as positive when generating labels for each frame
FRAME_LABEL_WINDOW_BEFORE_CRASH = 15
FRAME_LABEL_WINDOW_AFTER_CRASH = 30

NEIGHBORING_PREDICTION_FILTERING_THRESHOLD = 450  # Number of frames away to remove other predictions during filtering
PREDICTION_CORRECTNESS_THRESHOLD = 50  # Number of frames away for a prediction to be considered correct

MP4_EXT = '.mp4'

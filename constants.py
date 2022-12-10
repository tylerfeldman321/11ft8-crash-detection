DATA_DIR = 'data'
MODELS_DIR = 'models'
DEFAULT_PROBABILITY_THRESHOLD = 0.5

CRASH_FOLDER = 'data/crash_samples/'
SSIM_CSV = 'data/ssim.csv'
TIMESTAMPS_CSV = 'data/timestamps.csv'
LABELS_CSV = 'data/labels.csv'
AUDIO_CSV = 'data/audio.csv'
SIGN_DETECTION_VARIANCE_CSV = 'data/sign_detection_variance.csv'

TRAIN_PERCENTAGE = 0.75

FRAME_LABEL_WINDOW_BEFORE_CRASH = 15
FRAME_LABEL_WINDOW_AFTER_CRASH = 30

# TODO: make these in seconds instead of num frames
NEIGHBORING_PREDICTION_FILTERING_THRESHOLD = 450  # In number of frames
PREDICTION_CORRECTNESS_THRESHOLD = 50  # In num of frames

MP4_EXT = '.mp4'

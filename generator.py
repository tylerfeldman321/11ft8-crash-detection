import pandas as pd
import numpy as np
import os
from crash_bar_processing.crash_bar_ssim import CrashBarSSIM
from audio_processing.audio_processing import get_normalized_audio_amplitude
from sign_detection.sign_detector import SignDetector
from constants import CRASH_FOLDER, SSIM_CSV, TIMESTAMPS_CSV, LABELS_CSV, AUDIO_CSV, SIGN_DETECTION_VARIANCE_CSV
from constants import FRAME_LABEL_WINDOW_BEFORE_CRASH, FRAME_LABEL_WINDOW_AFTER_CRASH


def list_dir(rootdir):
    """ Creates a map from video folder (video name) to video filepath """
    ret = {}
    for vid_name in os.listdir(rootdir):
        d = os.path.join(rootdir, vid_name)
        if os.path.isdir(d):
            for vid in os.listdir(d):
                vid_file = os.path.join(d, vid)
                if os.path.isfile(vid_file) and vid.endswith('.copy.mp4'):
                    ret[vid_name] = vid_file
    return ret


def generate_ssim_data(crash_folder, csv_path):
    """ Create a pandas DataFrame where the row index is the frame and the column index is file. 
    The data is the ssim of each frame. Write it to a file after processing all videos. """
    videos = list_dir(crash_folder)
    detector = CrashBarSSIM()
    ssims = pd.DataFrame()
    for vid in videos:
        print(f'Calculating ssim for {vid}:')
        ssim, fps = detector.detect(videos[vid])
        ssims[vid] = pd.Series(ssim)
    print(ssims.head())
    ssims = ssims.fillna(1.0)
    ssims.to_csv(csv_path, index_label='frame')


def generate_audio_data(crash_folder, csv_path):
    """ Generate audio data from mp4 files """
    videos = list_dir(crash_folder)
    audio = pd.DataFrame()
    for vid in videos:
        print(f'Generating audio data for {vid}:')
        amplitude = get_normalized_audio_amplitude(videos[vid], vid)
        audio[vid] = pd.Series(amplitude)
    print(audio.head())
    audio = audio.fillna(0.0)
    audio.to_csv(csv_path, index_label='frame')


def generate_sign_detection_variance_data(crash_folder, csv_path):
    videos = list_dir(crash_folder)
    sign_detector = SignDetector()
    sign_detection_variance = pd.DataFrame()
    for vid in videos:
        print(f'Generating sign detection variance data for {vid}:')
        variance = sign_detector.process_video(videos[vid], compute_variance=True)
        sign_detection_variance[vid] = pd.Series(variance)
    print(sign_detection_variance.head())
    sign_detection_variance = sign_detection_variance.fillna(0.0)
    sign_detection_variance.to_csv(csv_path, index_label='frame')


def generate_labels(ssim_csv, timestamps_csv, labels_csv):
    ssims = pd.read_csv(ssim_csv, index_col='frame')
    timestamps = pd.read_csv(timestamps_csv)
    labels_all = pd.DataFrame()
    for file, frame in zip(timestamps['file'], timestamps['frame']):
        data = np.zeros(len(ssims))
        labels = pd.Series(data)
        labels.iloc[frame - FRAME_LABEL_WINDOW_BEFORE_CRASH:frame + FRAME_LABEL_WINDOW_AFTER_CRASH] = 1.0
        labels_all[file] = labels
    labels_all.to_csv(labels_csv, index_label='frame')


def generate_all_data():
    generate_labels(SSIM_CSV, TIMESTAMPS_CSV, LABELS_CSV)

    generate_ssim_data(CRASH_FOLDER, SSIM_CSV)
    generate_audio_data(CRASH_FOLDER, AUDIO_CSV)
    generate_sign_detection_variance_data(CRASH_FOLDER, SIGN_DETECTION_VARIANCE_CSV)


if __name__ == '__main__':
    generate_all_data()

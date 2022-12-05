import pandas as pd
import numpy as np
import os
from detector import Detector
from video_to_wav import generate_data

BEFORE_WINDOW = 15
AFTER_WINDOW = 30
CRASH_FOLDER = 'data/crash samples'
SSIM_CSV = 'data/ssim.csv'
TIMESTAMPS_CSV = 'data/timestamps.csv'
LABELS_CSV = 'data/labels.csv'
AUDIO_CSV = 'data/audio.csv'


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
    detector = Detector()
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
        amplitude = generate_data(videos[vid], vid)
        audio[vid] = pd.Series(amplitude)
    print(audio.head())
    audio = audio.fillna(0.0)
    audio.to_csv(csv_path, index_label='frame')


def generate_labels(ssim_csv, timestamps_csv, labels_csv):
    ssims = pd.read_csv(ssim_csv, index_col='frame')
    timestamps = pd.read_csv(timestamps_csv)
    labels_all = pd.DataFrame()
    for file, frame in zip(timestamps['file'], timestamps['frame']):
        data = np.zeros(len(ssims))
        labels = pd.Series(data)
        labels.iloc[frame - BEFORE_WINDOW:frame + AFTER_WINDOW] = 1.0
        labels_all[file] = labels
    labels_all.to_csv(labels_csv, index_label='frame')


def main():
    # Takes about 30 minutes to analyze all 10min videos
    # generate_ssim_data(CRASH_FOLDER, SSIM_CSV)
    # generate_labels(SSIM_CSV, TIMESTAMPS_CSV, LABELS_CSV)
    generate_audio_data(CRASH_FOLDER, AUDIO_CSV)


if __name__ == '__main__':
    main()

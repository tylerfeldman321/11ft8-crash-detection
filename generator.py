import pandas as pd
import numpy as np
import os
from detector import Detector


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
    ssims.to_csv(csv_path, index_label='frame')


def main():
	# Takes about 30 minutes to analyze all 10min videos
    generate_ssim_data('data/crash samples', 'data/ssim.csv')


if __name__ == '__main__':
    main()

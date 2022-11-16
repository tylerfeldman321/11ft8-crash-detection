import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import cv2
import numpy as np
from sign_detector import SignDetector
from utils import load_video, display_rectangle_of_image, get_date_of_video_file, display_video
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import glob
from sklearn.neighbors import KernelDensity


# TODO: average across all signs to get more generalizable template
# TODO: collect data for positive and negative examples and plot histogram of template matching values for the two distributions; see if they are linearly separable


DATA_DIR = '../data'
RESULTS_DIR = 'results'
LABELS_DIR = 'labels'


def plot_template_matching_for_video(self, video_path, skip=5, show=True, save=True):
    """ Plot template matching strength over time for a provided video """
    sign_detector = SignDetector()

    capture = load_video(video_path)
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_nums = []
    template_match_values = []

    for i in tqdm (range(num_frames), desc="Running template matching..."):
        ret, frame = capture.read()
        if frame is None:
            break
        if i % skip == 0:
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = sign_detector.get_roi_of_frame(gray_frame)
        max_val, max_loc = sign_detector.template_match(gray_frame)

        frame_nums.append(i)
        template_match_values.append(max_val)

    capture.release()
    frame_nums = np.array(frame_nums)
    template_match_values = np.array(template_match_values)

    date = get_date_of_video_file(video_path)
    plt.figure()
    plt.plot(frame_nums, template_match_values)
    plt.title(f'Template Matching Results for {date}')
    plt.xlabel('Frame')
    plt.ylabel('Strength of Template Matching')
    if save:
        plt.savefig(os.path.join(RESULTS_DIR, f'template_matching_{date}.png'))
    if show:
        plt.show()
    plt.close()

    save_path = os.path.join(RESULTS_DIR, f'template_matching_{date}')
    np.save(save_path, template_match_values)


def template_matching_for_all_videos_in_data(self):
    """ Save template matching results for all videos in data """
    video_paths = glob.glob(os.path.join(DATA_DIR, '*', '*.mp4'))
    for video_path in video_paths:
        plot_template_matching_for_video(video_path, show=False, save=True)


def save_image_and_template_from_video(video_path):
    """ Save image and template from wherever the video is stopped """
    date = get_date_of_video_file(video_path)
    last_frame = display_video(video_path)
    cv2.imwrite(os.path.join(RESULTS_DIR, f'saved_frame_{date}.png'), last_frame)
    template = get_sign_template_from_image(last_frame)
    cv2.imwrite(os.path.join(RESULTS_DIR, f'saved_template_{date}.png'), template)


def get_sign_template_from_image(image):
    """ Get sign template from an image """
    upper_left = (335, 890)
    lower_right = (405, 1020)
    return display_rectangle_of_image(image, upper_left, lower_right)


def label_data_from_video_file(video_path):
    """ Label data for a video file """
    sign_detector = SignDetector()

    capture = load_video(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = 0

    labels = np.zeros((num_frames))

    for i in tqdm (range(num_frames), desc="Running template matching..."):
        ret, frame = capture.read()
        if frame is None:
            break

        cv2.imshow("frame", cv2.resize(frame, (960, 540)))

        if skip == -1:
            label = 0
            cv2.waitKey(1)
        elif 0 < skip:
            label = 0
            skip -= 1
            cv2.waitKey(1)
        else:
            key = cv2.waitKey(0)
            label = 0  # If pressing a random key, then label will be false
            if key == ord('f'):  # Label frame as false example
                label = 0
            elif key == ord('t'):  # Label frame as true example
                label = 1
            elif key == ord('1'):  # Label the next 1 minutes as false examples
                skip = fps * 60
                label = 0
            elif key == ord('d'):  # Label rest of video as false examples
                skip = -1
                label = 0

        # Collect labels
        labels[i] = label

    date = get_date_of_video_file(video_path)
    save_path = os.path.join(LABELS_DIR, f'sign_labels_{date}')
    np.save(save_path, labels)


def plot_kde_and_roc(padding=10, n=10000):
    """ Plot KDE and ROC for labeled data """
    # TODO: why does p_d not reach 1?
    values, labels = load_labeled_data()

    negative_values = values[labels == 0].reshape(-1, 1)
    positive_values = values[labels == 1].reshape(-1, 1)

    minimum = min(negative_values.min(), positive_values.min())
    maximum = max(negative_values.max(), positive_values.max())

    kde_h0 = KernelDensity(kernel='gaussian').fit(negative_values)
    kde_h1 = KernelDensity(kernel='gaussian').fit(positive_values)

    s = np.linspace(minimum-padding, maximum+padding, n)
    dens_h0 = np.exp(kde_h0.score_samples(s.reshape(-1, 1)))
    plt.fill_between(s, dens_h0, alpha=0.3)

    dens_h1 = np.exp(kde_h1.score_samples(s.reshape(-1, 1)))
    plt.fill_between(s, dens_h1, alpha=0.3)

    plt.plot(s, dens_h0, label='Lambda Given H0')
    plt.plot(s, dens_h1, label='Lambda Given H1')
    plt.title(f'KDE Score vs. Lambda')
    plt.xlabel('Lambda')
    plt.ylabel('Kernel Density Estimation Score')
    plt.legend(loc='best')
    plt.show()

    p_fa_list = []
    p_d_list = []

    width = s[1] - s[0]
    for i in range(len(s)):
        p_fa = round(sum(dens_h0[i:]) * width, 4)
        p_d = round(sum(dens_h1[i:]) * width, 4)
        p_fa_list.append(p_fa)
        p_d_list.append(p_d)

    plt.plot(p_fa_list, p_d_list)
    plt.title('ROC Curve')
    plt.xlabel('Probability of False Alarm')
    plt.ylabel('Probability of Detection')
    plt.show()

    auc = 0
    for i in range(1, len(p_fa_list)):
        width = p_fa_list[i-1] - p_fa_list[i]
        height = (p_d_list[i] + p_d_list[i-1]) / 2
        auc += width * height
    print(f'Area under the curve: {auc}')


if __name__ == "__main__":

    # sign_on_example = cv2.imread('data/sign_on_example.png', 0)
    # template_match(sign_on_example, display_result=True)

    # plot_template_matching_for_video(r'data\2020-02-21_Penske-scrapes-roof-in-snow-c154\20200221.123001.11foot82b.copy.mp4')
    # template_matching_for_all_videos_in_data()

    # label_data_from_video_file(r'data\2019-10-03_Digger-hits-bridge-c148\20191003.141001.11foot82b.copy.mp4')
    

    # video_paths = glob.glob(os.path.join(DATA_DIR, '*', '*.mp4'))
    # for video_path in video_paths:
    #     print(video_path)
    #     save_image_and_template_from_video(video_path)

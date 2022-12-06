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
import pandas as pd


DATA_DIR = '../data'
RESULTS_DIR = 'results'
LABELS_DIR = 'labels'
TEMPLATE_DIR = 'templates'


def plot_template_matching_for_video(video_path, skip=5, show=True, save=True):
    """ Plot template matching strength over time for a provided video """
    sign_detector = SignDetector()

    basename = os.path.basename(os.path.dirname(video_path))

    capture = load_video(video_path)
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_nums = []
    template_match_values = []

    for i in tqdm (range(num_frames), desc="Running template matching..."):
        ret, frame = capture.read()
        if frame is None:
            break

        if i % skip == 0:
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
        plt.savefig(os.path.join(RESULTS_DIR, f'{basename}.png'))
    if show:
        plt.show()
    plt.close()

    save_path = os.path.join(RESULTS_DIR, f'{basename}')
    np.save(save_path, template_match_values)


def template_matching_for_all_videos_in_data():
    """ Save template matching results for all videos in data """
    video_paths = glob.glob(os.path.join(DATA_DIR, '*', '*.mp4'))
    for video_path in video_paths:
        plot_template_matching_for_video(video_path, show=False, save=True)


def save_image_and_template_from_video(video_path):
    """ Save image and template from wherever the video is stopped """
    date = get_date_of_video_file(video_path)
    last_frame = display_video(video_path)
    cv2.imwrite(os.path.join(TEMPLATE_DIR, f'saved_frame_{date}.png'), last_frame)
    template = get_sign_template_from_image(last_frame)
    cv2.imwrite(os.path.join(TEMPLATE_DIR, f'saved_template_{date}.png'), template)


def get_sign_template_from_image(image):
    """ Get sign template from an image """
    # Templates so far are 70 pixels tall, 130 pixels wide
    upper_left = (230, 895)  # (335, 890) for anything before and including 20210729
    lower_right = (300, 1025)  # (405, 1020) for anything before and including 20210729
    template = display_rectangle_of_image(image, upper_left, lower_right)
    return template


def get_average_template():
    templates = glob.glob(os.path.join(TEMPLATE_DIR, 'saved_template_*.png'))
    num_templates = len(templates)
    print(f'Found {num_templates} templates')

    avg_template = None
    for template in templates:
        template_np = cv2.imread(template)
        if avg_template is None:
            avg_template = template_np / num_templates
        else:
            avg_template += (template_np / num_templates)
    avg_template_save_path = os.path.join(TEMPLATE_DIR, 'avg_template.png')
    cv2.imwrite(avg_template_save_path, avg_template)
    print(f'Saved average template to {avg_template_save_path}')


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
    save_path = os.path.join(LABELS_DIR, os.path.basename(os.path.dirname(video_path)))
    np.save(save_path, labels)


def plot_kde_and_roc(padding=0.5, n=10000):
    """ Plot KDE and ROC for labeled data """
    values, labels = load_labeled_data()

    negative_values = values[labels == 0].reshape(-1, 1)
    positive_values = values[labels == 1].reshape(-1, 1)

    assert positive_values.shape[0] != 0, 'No positive examples in the provided data'

    minimum = min(negative_values.min(), positive_values.min())
    maximum = max(negative_values.max(), positive_values.max())

    kde_h0 = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(negative_values)
    kde_h1 = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(positive_values)

    s = np.linspace(minimum-padding, maximum+padding, n)
    dens_h0 = np.exp(kde_h0.score_samples(s.reshape(-1, 1)))
    plt.fill_between(s, dens_h0, alpha=0.3)

    dens_h1 = np.exp(kde_h1.score_samples(s.reshape(-1, 1)))
    plt.fill_between(s, dens_h1, alpha=0.3)

    plt.plot(s, dens_h0, label='Negative Examples')
    plt.plot(s, dens_h1, label='Positive Examples')
    plt.title(f'KDE Score vs. Normalized Cross Correlation Value')
    plt.xlabel('Normalized Cross Correlation Value')
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
    plt.title('ROC Curve for Sign Detection')
    plt.xlabel('Probability of False Alarm')
    plt.ylabel('Probability of Detection')
    plt.show()

    auc = 0
    for i in range(1, len(p_fa_list)):
        width = p_fa_list[i-1] - p_fa_list[i]
        height = (p_d_list[i] + p_d_list[i-1]) / 2
        auc += width * height
    print(f'Area under the curve: {auc}')


def load_labeled_data():
    """ Loads labeled data """

    label_files = glob.glob(os.path.join(LABELS_DIR, "*.npy"))

    aggregated_values = None
    aggregated_labels = None

    for label_file in label_files:
        labels = np.load(label_file)
        labels = clean_and_pad_sign_detection_results(labels)

        date = os.path.splitext(os.path.basename(label_file))[0].split('_')[-1]
        values_path = os.path.join(RESULTS_DIR, os.path.basename(label_file))

        if not os.path.exists(values_path):
            print(f'{values_path} does not exist, skipping this set of data')
            continue

        values = np.load(values_path)
        values = clean_and_pad_sign_detection_results(values)

        if aggregated_values is None:
            aggregated_values = values
        else:
            aggregated_values = np.hstack((aggregated_values, values))

        if aggregated_labels is None:
            aggregated_labels = labels
        else:
            aggregated_labels = np.hstack((aggregated_labels, labels))

    return aggregated_values, aggregated_labels


def convert_data_to_csv(num_frames=9000, skip=5):
    """ Converts the .npy files stored after running template_matching_for_all_videos_in_data()
    into a CSV file with the proper format """
    df_dict = {}

    df_dict["frame"] = np.arange(0, num_frames, 1)

    sign_detection_results_files = glob.glob(os.path.join(RESULTS_DIR, "*.npy"))

    for sign_detection_results_file in sign_detection_results_files:

        sign_detection_results = np.load(sign_detection_results_file)
        sign_detection_results = clean_and_pad_sign_detection_results(sign_detection_results, num_frames, skip)

        filename = os.path.splitext(os.path.basename(sign_detection_results_file))[0]

        df_dict[f'{filename}'] = list(sign_detection_results)

    df = pd.DataFrame(df_dict)
    df.to_csv(os.path.join(DATA_DIR, 'sign_detection_results.csv'), index=False)


def clean_and_pad_sign_detection_results(sign_detection_results, num_frames=9000, skip=5):
    sign_detection_results = np.repeat(sign_detection_results, skip)
    sign_detection_results = sign_detection_results[0:num_frames]

    if len(sign_detection_results) < num_frames:
        num_missing_frames = num_frames - len(sign_detection_results)
        last_val = sign_detection_results[-1]
        array_to_append = np.repeat(np.array([last_val]), num_missing_frames)
        sign_detection_results = np.append(sign_detection_results, array_to_append)

    return sign_detection_results


def automatic_data_label():
    for file in glob.glob(os.path.join(RESULTS_DIR, "*.npy")):
        print(file)
        dataarray = np.load(file)

        plt.clf()
        plt.figure(1)
        plt.plot(dataarray)
        plt.title(file)
        plt.xlabel('Frame')
        plt.show()
        plt.close()

        threshold = input("Treshold: ")

        if threshold == '':
            print('Skipped')
            continue

        threshold = float(threshold)
        labels = (dataarray > threshold) * 1
        np.save(os.path.join(LABELS_DIR, os.path.basename(file)), labels)

        plt.clf()
        plt.figure(2)
        plt.title(file)
        plt.plot(labels)
        plt.xlabel('Frame')
        plt.show()
        plt.close()
            

if __name__ == "__main__":
    # plot_template_matching_for_video(r'..\data\2019-12-19_Lost-cargo-evening-light-c152\20191219.125001.11foot82b.copy.mp4')
    # template_matching_for_all_videos_in_data()

    # label_data_from_video_file(r'..\data\2019-12-19_Lost-cargo-evening-light-c152\20191219.125001.11foot82b.copy.mp4')
    # automatic_data_label()
    plot_kde_and_roc()

    # video_paths = glob.glob(os.path.join(DATA_DIR, '*', '*.mp4'))
    # for video_path in video_paths:
    #     print(video_path)
    #     save_image_and_template_from_video(video_path)

    
    # get_average_template()

    # convert_data_to_csv()

    pass

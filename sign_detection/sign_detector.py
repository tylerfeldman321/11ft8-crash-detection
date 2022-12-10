import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob

DATA_DIR = '../data'
RESULTS_DIR = 'results'
LABELS_DIR = 'labels'
TEMPLATE_DIR = 'templates'
TEMPLATE_FILEPATH = os.path.join('sign_detection', TEMPLATE_DIR, 'sign_on_template.png')

TEMPLATE_MATCHING_METHODS = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
                             cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
TM_CCOEFF_THRESHOLD = 1000000


class SignDetector:

    Y_MIN = 250
    Y_MAX = 500
    X_MIN = 750
    X_MAX = 1200

    def __init__(self):
        self.template = self._load_template(os.path.join(os.getcwd(), TEMPLATE_FILEPATH))

    def is_sign_on(self, img):
        max_val, max_loc = self.template_match(img)

    def template_match(self, img, method=cv2.TM_CCOEFF_NORMED, display_result=False):
        """ Apply template matching to the input image """
        h, w = self.template.shape
        res = cv2.matchTemplate(img, self.template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if display_result:
            self._display_template_match_result(img, res, min_loc, max_loc, method, w, h)
        return max_val, max_loc

    def process_video(self, video_filepath, compute_variance=True, skip=5):
        capture = cv2.VideoCapture(video_filepath)
        num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        template_match_values = []

        for i in tqdm(range(num_frames), f'Running template matching: '):
            ret, frame = capture.read()
            if frame is None:
                break

            if i % skip == 0:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_frame = self.get_roi_of_frame(gray_frame)
                max_val, max_loc = self.template_match(gray_frame)

                template_match_values.append(max_val)

        template_match_values = np.array(template_match_values)

        capture.release()
        template_match_values = clean_and_pad_sign_detection_results(
            template_match_values, num_frames=num_frames)

        if compute_variance:
            variance = extract_variance_of_moving_window(template_match_values)
            return variance
        else:
            return template_match_values

    def _display_template_match_result(self, img, res, min_loc, max_loc, method, w, h):
        """ Display results of template matching """
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img, top_left, bottom_right, 255, 2)

        plt.subplot(121)
        plt.imshow(res, cmap='gray')
        plt.title('Template Matching Result')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img, cmap='gray')
        plt.title('Detected Point')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def _load_template(self, path=TEMPLATE_FILEPATH):
        """ Load template as grayscale imgae """
        sign_on_template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return sign_on_template

    def get_roi_of_frame(self, frame):
        frame = frame[self.Y_MIN:self.Y_MAX, self.X_MIN:self.X_MAX]
        return frame


def extract_variance_of_moving_window(sign_detection_results, window_size=1000):
    return extract_moving_window_function(sign_detection_results, func=np.var, window_size=window_size)


def extract_moving_window_function(sign_detection_results, func=np.var, window_size=1000):
    results = np.zeros(sign_detection_results.shape)
    for i, sign_detection_val in enumerate(sign_detection_results):
        if i - window_size < 0:
            val = func(sign_detection_results[0:i+1])
        else:
            val = func(sign_detection_results[i-window_size:i+1])
        results[i] = val
    return results


def clean_and_pad_sign_detection_results(sign_detection_results, num_frames=9000, skip=5):
    sign_detection_results = np.repeat(sign_detection_results, skip)
    sign_detection_results = sign_detection_results[0:num_frames]

    if len(sign_detection_results) < num_frames:
        num_missing_frames = num_frames - len(sign_detection_results)
        last_val = sign_detection_results[-1]
        array_to_append = np.repeat(np.array([last_val]), num_missing_frames)
        sign_detection_results = np.append(sign_detection_results, array_to_append)

    return sign_detection_results


if __name__ == '__main__':
    sd = SignDetector()
    img = cv2.imread(os.path.join(DATA_DIR, 'sign_on_example.png'), cv2.IMREAD_GRAYSCALE)
    sd.template_match(sd.get_roi_of_frame(img), display_result=True)

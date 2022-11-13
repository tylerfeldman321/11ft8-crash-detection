import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import load_video
from tqdm import tqdm
import glob

# TODO: average across all signs to get more generalizable template
# TODO: collect data for positive and negative examples and plot histogram of template matching values for the two distributions; see if they are linearly separable

DATA_DIR = '../data'
TEMPLATE_FILEPATH = os.path.join(DATA_DIR, 'sign_on_template.png')
TEMPLATE_MATCHING_METHODS = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
                             cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
TM_CCOEFF_THRESHOLD = 1000000

class SignDetector:

    Y_MIN = 250
    Y_MAX = 500
    X_MIN = 750
    X_MAX = 1200

    def __init__(self):
        pass

    def is_sign_on(self, img):
        max_val, max_loc = self.template_match(img)

    def template_match(self, img, method=TEMPLATE_MATCHING_METHODS[0], display_result=False):
        """ Apply template matching to the input image """
        template = self._load_template()
        h, w = template.shape
        res = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if display_result:
            self._display_template_match_result(img, res, min_loc, max_loc, method, w, h)
        return max_val, max_loc

    def _display_template_match_result(self, img, res, min_loc, max_loc, method, w, h):
        """ Display results of template matching """
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img, top_left, bottom_right, 255, 2)

        plt.subplot(121)
        plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img,cmap = 'gray')
        plt.title('Detected Point')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def _load_template(self):
        """ Load template as grayscale imgae """
        sign_on_template = cv2.imread(TEMPLATE_FILEPATH, cv2.IMREAD_GRAYSCALE)
        return sign_on_template

    def get_roi_of_frame(self, frame):
        frame = frame[self.Y_MIN:self.Y_MAX, self.X_MIN:self.X_MAX]
        return frame

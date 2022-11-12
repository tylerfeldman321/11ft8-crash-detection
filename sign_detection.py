import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from utils import load_video
from tqdm import tqdm


TEMPLATE_FILEPATH = os.path.join('data', 'sign_on_template.png')
TEMPLATE_MATCHING_METHODS = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
                             cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
TM_CCOEFF_THRESHOLD = 1000000

class SignDetector:

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

    def plot_template_matching_for_video(self, video_path, skip=5):
        """ Plot template matching strength over time for a provided video """
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
            max_val, max_loc = self.template_match(gray_frame)

            frame_nums.append(i)
            template_match_values.append(max_val)

        capture.release()
        frame_nums = np.array(frame_nums)
        template_match_values = np.array(template_match_values)

        video_file_basename = os.path.basename(video_path)
        plt.plot(frame_nums, template_match_values)
        plt.title(f'Template Matching Results for {video_file_basename}')
        plt.xlabel('Frame')
        plt.ylabel('Strength of Template Matching')
        plt.show()

        np.save(f'template_matching_{video_file_basename}', template_match_values)


if __name__ == "__main__":
    sign_detector = SignDetector()

    sign_on_example = cv2.imread('data/sign_on_example.png', 0)
    sign_detector.template_match(sign_on_example, display_result=True)

    sign_detector.plot_template_matching_for_video('data/20200103.082000.11foot82b.copy.mp4')

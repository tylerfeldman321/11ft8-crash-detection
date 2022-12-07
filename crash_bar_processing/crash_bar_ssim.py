import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from tqdm import tqdm


class CrashBarSSIM:

    LOWER_YELLOW = np.array([10, 110, 110])  # HSL
    UPPER_YELLOW = np.array([70, 255, 255])  # HSL
    # Expected dimensions of crash bar
    THRESHOLD_HEIGHT = 15
    THRESHOLD_WIDTH = 500
    REFERENCE_IMAGE = 'data/reference.jpg'

    def __init__(self):
        """ Set default image bounds """
        self.y_min = 0
        self.y_max = 1080
        self.x_min = 0
        self.x_max = 1920

    def detect(self, file, visualize=False):
        """ Detect the likelihood of a crash in an mp4 file.
        Returns likelihood of crash for each frame in the video, between 0 and 1.
        Lower values indicate higher crash likelihoods. """
        first_frame = self._get_first_frame(file)
        bar_mask = self._find_crash_bar(first_frame)
        # self._play_video(file)
        differences, fps = self._process_video(bar_mask, first_frame, file)
        if visualize:
            self._visualize_differences(differences, file, fps=fps)
        return np.array(differences), fps

    def _play_video(self, file, fps=15):
        cap = cv2.VideoCapture(file)
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:  # End of video
                break
            cv2.imshow('video', frame)
            if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):  # Press q to quit
                break

    def _process_video(self, mask, target, file):
        """ Return a time series with the difference between the original bar and any given frame"""
        masked_target = cv2.bitwise_and(target, target, mask=mask)
        cap = cv2.VideoCapture(file)
        differences = []
        pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:  # End of video
                break
            pbar.update(1)
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
            differences.append(self._calc_difference(masked_frame, masked_target))
            masked_target = masked_frame  # Compare to previous frame
        cap.release()
        return differences, fps

    def _calc_difference(self, frame, target):
        """ Calculate the structural similarity between the frame and the target. """
        t = target[self.y_min:self.y_max, self.x_min:self.x_max]
        f = frame[self.y_min:self.y_max, self.x_min:self.x_max]
        return structural_similarity(f, t, channel_axis=2)

    def _get_first_frame(self, file):
        """ Get first frame of video """
        cap = cv2.VideoCapture(file)
        success, image = cap.read()
        # cv2.imwrite('data/first_frame.jpg', image)
        cap.release()
        if success:
            return image
        raise Exception('Could not read first frame.')

    def _find_crash_bar(self, image):
        """ Find the crash bar in the image. First, use color masking and contour detection to find
        a rotated rectangle that bounds the crash bar. Then, calculate a rectangular mask containing
        the crash bar.
        """
        try:
            color_mask, contour, rect = self._calc_color_mask(image)
            print('Found crash bar.')
        except Exception as e:
            print('Could not find crash bar, using crash bar location in reference image.')
            reference = cv2.imread(self.REFERENCE_IMAGE)
            color_mask, contour, rect = self._calc_color_mask(reference)

        rect_mask = self._calc_rect_mask(image, contour)
        mask = color_mask & rect_mask
        # self._display_mask(mask, image)
        res = cv2.bitwise_and(image, image, mask=mask)
        # cv2.imwrite('images/final-mask.jpg', res)
        return mask

    def _calc_color_mask(self, image):
        """ Masks the image by color and finds the rotated rectangle that bounds the crash bar """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert image to HSV
        color_mask = cv2.inRange(hsv, self.LOWER_YELLOW, self.UPPER_YELLOW)
        # self._display_mask(color_mask, image)
        res = cv2.bitwise_and(image, image, mask=color_mask)
        # cv2.imwrite('images/color-mask.jpg', res)

        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # cv2.imwrite('images/good.jpg', image)
        # self._display_contours(res, sorted_contours)

        for contour in sorted_contours:
            rect = cv2.minAreaRect(contour)
            if rect[1][0] > self.THRESHOLD_WIDTH and rect[1][1] > self.THRESHOLD_HEIGHT:
                # self._display_bounding_box(image, rect)
                return color_mask, contour, rect
        raise Exception('Could not find crash bar.')

    def _calc_rect_mask(self, image, contour):
        """ Calculate the rectangular bounding box of the crash bar from a contour."""
        (x, y, w, h) = cv2.boundingRect(contour)

        self.x_min = x
        self.x_max = x + w
        self.y_min = y
        self.y_max = y + h

        rect_mask = np.zeros(image.shape[:2], np.uint8)
        rect_mask[self.y_min:self.y_max, self.x_min:self.x_max] = 255
        return rect_mask

    def _display_contours(self, image, contours):
        disp = image.copy()
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(disp, [box], 0, (0, 0, 255), 2)
        cv2.imshow('contours', disp)
        # cv2.imwrite('images/good-contours.jpg', disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _display_bounding_box(self, image, rect):
        disp = image.copy()
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(disp, [box], 0, (0, 0, 255), 2)
        cv2.imshow('crash_bar', disp)
        # cv2.imwrite('images/contour.jpg', disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _display_mask(self, mask, image):
        """ Display the results of masking an image """
        res = cv2.bitwise_and(image, image, mask=mask)

        cv2.imshow('image', image)
        cv2.imshow('mask', mask)
        cv2.imshow('res', res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _visualize_differences(self, diff, file, fps=15):
        """ Plots crash likelihood for each frame """
        timestamps = [(1/fps) * i for i in range(len(diff))]
        # normal = (diff-np.min(diff))/(np.max(diff)-np.min(diff))
        plt.plot(timestamps, diff)
        plt.title(f'Framewise Structural Similarity in {file}')
        plt.xlabel('Time (s)')
        plt.ylabel('Structural Similarity')
        plt.show()


def main(args):
    detector = CrashBarSSIM()
    detector.detect(args.file, visualize=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect pixel changes in a video file')
    parser.add_argument('file', help='Path to video file')
    args = parser.parse_args()
    main(args)

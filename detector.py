import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Detector:

    LOWER_YELLOW = np.array([10, 110, 110])  # HSL
    UPPER_YELLOW = np.array([70, 255, 255])  # HSL
    # Expected dimensions of crash bar
    THRESHOLD_HEIGHT = 15
    THRESHOLD_WIDTH = 500
    REFERENCE_IMAGE = 'data/reference.jpg'

    def __init__(self):
        self.y_min = 0
        self.y_max = 1080
        self.x_min = 0
        self.x_max = 1920

    def detect(self, file):
        """ Detect the likelihood of a crash in an mp4 file.
        Returns a likelihood of crash for each frame in the video, normalized from 0 to 1 """
        first_frame = self._get_first_frame(file)
        bar_mask = self._find_crash_bar(first_frame)
        # self._play_video(file)
        differences = self._process_video(bar_mask, first_frame, file)
        self._visualize_differences(differences)

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
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:  # End of video
                break
            pbar.update(1)
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
            differences.append(self._calc_difference(masked_frame, masked_target))
            masked_target = masked_frame  # Compare to previous frame
        cap.release()
        return differences

    def _calc_difference(self, frame, target):
        """ Calculate the normalized cross-correlation between the frame and the target. """
        # Slicing images for faster calculations (huge speedup!)
        t = target[self.y_min:self.y_max, self.x_min:self.x_max]
        f = frame[self.y_min:self.y_max, self.x_min:self.x_max]
        dist_ncc = np.sum((f - np.mean(f)) * (t - np.mean(t))) / \
            ((f.size - 1) * np.std(f) * np.std(t))
        return dist_ncc

    def _visualize_differences(self, diff, fps=15):
        """ Likelihood of crash for a frame is defined as the normalized difference between the 
        frame and the previous frame. See _calc_difference for how this difference is calculated.
        """
        timestamps = [(1/fps) * i for i in range(len(diff))]
        normal = (diff-np.min(diff))/(np.max(diff)-np.min(diff))
        plt.plot(timestamps, normal)
        plt.title("Normalized Cross Correlation of Video")
        plt.xlabel("Time (s)")
        plt.ylabel("Normalized Cross Correlation")
        plt.show()

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
        except Exception as e:
            print('Could not find crash bar, falling back to reference image.')
            reference = cv2.imread(self.REFERENCE_IMAGE)
            color_mask, contour, rect = self._calc_color_mask(reference)

        rect_mask = self._calc_rect_mask(image, contour)
        mask = color_mask & rect_mask
        # self._display_mask(mask, image)
        return mask

    def _calc_color_mask(self, image):
        """ Masks the image by color and finds the rotated rectangle that bounds the crash bar """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert image to HSV
        color_mask = cv2.inRange(hsv, self.LOWER_YELLOW, self.UPPER_YELLOW)

        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

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

    def _display_bounding_box(self, image, rect):
        disp = image.copy()
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(disp, [box], 0, (0, 0, 255), 2)
        cv2.imshow('crash_bar', disp)
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


def main(args):
    detector = Detector()
    detector.detect(args.file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detect pixel changes in a video file")
    parser.add_argument('file', help='Path to video file')
    args = parser.parse_args()
    main(args)

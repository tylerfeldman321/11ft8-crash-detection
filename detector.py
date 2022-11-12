import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Detector:

    LOWER_YELLOW = np.array([0, 70, 180])  # sRGB
    UPPER_YELLOW = np.array([30, 120, 255])  # sRGB
    Y_MIN = 440
    Y_MAX = 500
    X_MIN = 740
    X_MAX = 1340
    THRESHOLD = 100  # This is a hyperparameter that needs to be tuned

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
        """ Calculate the difference between the frame and the target.
        TODO: Explore different algorithms here. """
        # Slicing images for faster calculations (3x speedup!)
        sliced_target = target[self.Y_MIN:self.Y_MAX, self.X_MIN:self.X_MAX]
        sliced_frame = frame[self.Y_MIN:self.Y_MAX, self.X_MIN:self.X_MAX]
        # Count how many channels differ by a certain threshold
        return np.sum(np.square(np.subtract(sliced_frame, sliced_target)) > self.THRESHOLD)

    def _visualize_differences(self, diff, fps=15):
        """ Likelihood of crash for a frame is defined as the normalized difference between the 
        frame and the previous frame. See _calc_difference for how this difference is calculated.
        """
        timestamps = [(1/fps) * i for i in range(len(diff))]
        normal = (diff-np.min(diff))/(np.max(diff)-np.min(diff))
        plt.plot(timestamps, normal)
        plt.title("Likelihood of Crash")
        plt.xlabel("Time (s)")
        plt.ylabel("Likelihood")
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
        """ Naively find the crash bar in the image. Assumes the bar is all of the 'yellow' pixels
        within a pre-defined region. Return a mask corresponding to the bar location.
        """
        rect_mask = np.zeros(image.shape[:2], np.uint8)
        rect_mask[self.Y_MIN:self.Y_MAX, self.X_MIN:self.X_MAX] = 255
        color_mask = cv2.inRange(image, self.LOWER_YELLOW, self.UPPER_YELLOW)
        mask = color_mask & rect_mask
        # self._display_mask(mask, image)
        return mask

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

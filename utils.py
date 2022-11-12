import cv2
import numpy as np


def load_video(filepath):
    """ Load video as a cv2 VideoCapture given its filepath """
    capture = cv2.VideoCapture(filepath)
    return capture


def display_video(filepath, frame_shape=(960, 540)):
    """ Display a video given its filepath and desired frame size """
    capture = cv2.VideoCapture(filepath)
    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        frame_resized = cv2.resize(frame, frame_shape)
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)

        if cv2.waitKey(1) == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def load_video(filepath):
    """ Load video as a cv2 VideoCapture given its filepath """
    cap = cv2.VideoCapture(filepath)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    print(f'Loaded video from {filepath}, with {num_frames} frames, {width} width, {height} height, and {fps:.3} fps')
    return cap


def display_video(filepath, frame_shape=(960, 540)):
    """ Display a video given its filepath and desired frame size. Return last frame shown """
    capture = cv2.VideoCapture(filepath)
    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        frame_resized = cv2.resize(frame, frame_shape)
        cv2.imshow('frame', frame_resized)

        if cv2.waitKey(1) == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
    return frame


def display_rectangle_of_image(image, upper_left=(250, 750), lower_right=(500, 1200)):
    """ Display a portion of an image defined by the upper left and lower right corners.
    The corners are in units of pixels in format row, col. image can be either a numpy array
    or a path to an image """

    if type(image) == str:
        image = cv2.imread(image)

    image_rectangle = image[upper_left[0]:lower_right[0],upper_left[1]:lower_right[1],:]
    plt.imshow(image_rectangle)
    plt.show()
    return image_rectangle


def get_date_of_video_file(video_path):
    video_file_basename = os.path.basename(video_path)
    date = video_file_basename.split('.')[0]
    return date

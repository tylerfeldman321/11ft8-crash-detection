import cv2
import numpy as np
import matplotlib.pyplot as plt


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
    """ Display a video given its filepath and desired frame size """
    capture = cv2.VideoCapture(filepath)
    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_resized = cv2.resize(gray, frame_shape)
        cv2.imshow('frame', frame_resized)

        if cv2.waitKey(1) == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


def display_rectangle_of_image(image_path, upper_left=(250, 750), lower_right=(500, 1200)):
    """ Display a portion of an image defined by the upper left and lower right corners.
    The corners are in units of pixels in format row, col """
    image = cv2.imread(image_path)
    image_rectangle = image[upper_left[0]:lower_right[0],upper_left[1]:lower_right[1],:]
    plt.imshow(image_rectangle)
    plt.show()

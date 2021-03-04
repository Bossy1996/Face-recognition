import PIL.Image
import numpy as np
from PIL import ImageFile
import dlib

try:
    import dlib_models_face_recognition
except Exception:
    print("Please install `dlib-models-face-recognition` with this command before using `Face_recognition`:\n`")
    print("pip install git+https://github.com/Bossy1996/dlib-models-face-recognition")
    quit()

ImageFile.LOAD_TRUNCATED_IMAGES = True

face_detector = dlib.get_frontal_face_detector()

predictor_68_point_model = dlib_models_face_recognition.pose_predictor_model_location()
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

predictor_5_point_model = dlib_models_face_recognition.pose_predictor_five_point_model_location()
pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)

cnn_face_detection_model = dlib_models_face_recognition.cnn_face_detector_model_location()
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)

face_recognition_model = dlib_models_face_recognition.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)


def load_image_file(file, mode='RGB'):
    """
    Loads an image file (.jpg, png, etc...) into numpy array
    :param file: image file name or file object to load
    :param mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white) are supported.
    :return: image content as a numpy array
    """
    img = PIL.Image.open(file)

    if mode:
        img = img.convert(mode)

    return np.array(img)


def face_locations(img, number_of_times_to_upsample=1, model='hog'):
    """
    Returns an array of bounding boxes of human faces in a image

    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces
    :param model: Which face model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate
        deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog",
    :return: A list of tuples of found face locations in css (top, right, bottom, left) order
    """
    if model == "cnn":
        return [_trim_css_to_bounds(_rect_to_css(face.rect), img.shape) for face in
                _raw_face_locations(img, number_of_times_to_upsample, "cnn")]
    else:
        return [_trim_css_to_bounds(_rect_to_css(face.rect), img.shape) for face in
                _raw_face_locations(img, number_of_times_to_upsample, model)]


def _raw_face_locations():
    pass


def _trim_css_to_bounds():
    pass


def _rect_to_css():
    pass

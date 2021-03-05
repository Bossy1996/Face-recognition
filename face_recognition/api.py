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


def face_encodings(face_image, known_face_locations=None, num_jitters=1, model="small"):
    """
    Given an image, return the 128-dimension face encoding for each face in the image.

    :param face_image: The image that contains one or more faces
    :param known_face_locations: Optional - the bounding
    boxes of each face if you already know them.
    :param num_jitters: How many times to re-sample the face when
    calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
    :param model: Optional - which model to use. "large" or "small" (default) which only returns 5 point but is faster.
    :return: A list of 128-dimensional face encodings (one for each face in the image)
    """
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations, model)
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for
            raw_landmark_set in raw_landmarks]


def _raw_face_locations(img, number_of_times_upsample=1, model="hog"):
    """
    Returns an array of bounding boxes of human faces in a image.

    :param img: An image (as a numpy array)
    :param number_of_times_upsample: How many times to upsample the image looking for faces. Higher numbers find smaller faces.
    :param model: Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is more accurate
    deep-learning model which is GPU/CUDA accelerated (if available). The default is "hog".
    :return: A list of dlib 'rect' objects of found face locations
    """
    if model == 'cnn':
        return cnn_face_detector(img, number_of_times_upsample)
    else:
        return face_detector(img, number_of_times_upsample)


def _rect_to_css(rect):
    """
    Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order

    :param rect: a dlib 'rect' object.
    :return: a plain tuple representation of the rect in (top, right , bottom, left) order
    """
    return rect.top(), rect.right(), rect.bottom(), rect.left()


def _css_to_rect(css):
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib 'rect' object.

    :param css: plain tuple representation of the rect in (top, right, bottom, left) order
    :return: a dlib 'rect' object
    """
    return dlib.rectangle(css[3], css[0], css[1], css[2])


def _trim_css_to_bounds(css, image_shape):
    """
    Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.

    :param css: plain tuple representation of the rect in (top, right, bottom, left) order.
    :param image_shape: numpy shape of the image array
    :return: a trimed plain tuple representation of the rect in (top, right, bottom, left) order.
    """
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)


def face_distance(face_encodings, face_to_compare):
    pass


def _raw_face_locations_batched(images, number_of_times_to_upsamples=1, batch_size=128):
    pass


def batch_face_locations(images, number_of_tiumes_to_upsample=1, batch_size=128):
    pass


def _raw_face_landmarks(face_image, face_locations=None, model="large"):
    pass


def face_landmarks(face_image, face_loactions=None, model="large"):
    pass


def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    pass

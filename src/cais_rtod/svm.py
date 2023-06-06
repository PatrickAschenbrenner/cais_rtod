import cv2
import numpy as np
from hog import get_img_feature


def svm_model(c: float = 10.0,
              gamma: float = 10.0) -> cv2.ml_SVM:
    """
    Create a SVM model which can be used to train and predict.
    """
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(c)
    svm.setGamma(gamma)
    svm.setKernel(cv2.ml.SVM_CHI2)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 10000, 1e-7))
    return svm


def predict(image,
            hog: cv2.HOGDescriptor,
            svm: cv2.ml_SVM,
            source: str = 'file') -> int:
    """
    Predict the label of the image. +1 if there is a hand, -1 otherwise.
    :param image:      image location or image object
    :param hog:        HOG descriptor
    :param svm:        SVM model
    :param source:     type of image (file or array)
    :return:           label of the image
    """
    image_feature = get_img_feature(image, hog, source).reshape(1, -1).astype(np.float32)
    _, result = svm.predict(image_feature)
    return result[0][0]


def save_svm(svm: cv2.ml_SVM,
             save_path: str):
    """
    Save a trained SVM model to the given path.
    :param svm:       SVM model
    :param save_path: path to save the model
    """
    if save_path[-4:] != ".xml":
        save_path += ".xml"
    svm.save(save_path)


def load_svm(load_path: str) -> cv2.ml_SVM:
    """
    Load a trained SVM model from the given path.
    :param load_path: path to load the model
    :return:          SVM model
    """
    if load_path[-4:] != ".xml":
        load_path += ".xml"
    svm = cv2.ml.SVM_load(load_path)
    return svm

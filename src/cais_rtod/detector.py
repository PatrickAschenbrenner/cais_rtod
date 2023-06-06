import svm
import hog
import yolo
import os
import numpy as np

package_directory = os.path.dirname(os.path.abspath(__file__))

class SVM():
    def __init__(self) -> None:
        self.model_path = os.path.join(package_directory, 'models', 'svm', 'svm.xml')
        self.svm_detector = svm.load_svm(self.model_path)
        self.hog_descriptor = hog.hog_descriptor()

    def predict(self, image) -> int:
        """
        Predict the label of the image. +1 if there is a hand, -1 otherwise.
        :param image:      image location or image object
        :return:           label of the image
        """
        if type(image) == str:
            source = 'file'
        elif type(image) == np.ndarray:
            source = 'array'
        else:
            raise ValueError('Image type not supported')
        return svm.predict(image, self.hog_descriptor, self.svm_detector, source)

class YOLOv3():
    def __init__(self) -> None:
        self.model_path = os.path.join(package_directory, 'models', 'yolov3', 'tiny-yolov3.pt')
        self.json_path = os.path.join(package_directory, 'models', 'yolov3', 'tiny-yolov3.json')
        self.yolo_detector = yolo.load_yolov3_detector(self.model_path, self.json_path)
    
    def predict(self, image) -> int:
        """
        Predict the label of the image. +1 if there is a hand, -1 otherwise.
        :param image:      image location or image object
        :return:           label of the image
        """
        _, detections = self.yolo_detector.detectObjectsFromImage(input_image=image, output_type="array")
        if len(detections) > 0:
            return 1
        else:
            return -1
    
    def predict_boxes(self, image) -> list:
        """
        Return a list of the bounding boxes for detected hands.
        Each list element is a tuple in YOLO format + confidence score:
        (x_center, y_center, width, height, confidence).
        :param image:      image location or image object
        :return:           list of bounding boxes
        """
        _, detections = self.yolo_detector.detectObjectsFromImage(input_image=image, output_type="array")
        if len(detections) == 0:
            return []
        return None

class YOLOv8():
    def __init__(self) -> None:
        self.model_path = os.path.join(package_directory, 'models', 'yolov8', 'small-yolov8.pt')
        self.yolo_detector = yolo.load_yolov3_detector(self.model_path, self.json_path)
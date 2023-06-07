from cais_rtod import svm
from cais_rtod import hog
from cais_rtod import yolo
import os
import numpy as np

package_directory = os.path.dirname(os.path.abspath(__file__))

class SVM():
    def __init__(self, custom_model = None) -> None:
        self.model_path = os.path.join(package_directory, 'models', 'svm', 'svm.xml')
        if custom_model:
            self.model_path = custom_model
        self.svm_detector = svm.load_svm(self.model_path)
        self.hog_descriptor = hog.hog_descriptor()

    def predict(self, image) -> int:
        """
        Predict the label of the image. +1 if there is a hand, -1 otherwise.
        :param image:      image location or ndarray (BGR)
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
    def __init__(self, custom_model = None, custom_json = None) -> None:
        self.model_path = os.path.join(package_directory, 'models', 'yolov3', 'tiny-yolov3.pt')
        self.json_path = os.path.join(package_directory, 'models', 'yolov3', 'tiny-yolov3.json')
        if custom_model:
            self.model_path = custom_model
        if custom_json:
            self.json_path = custom_json
        self.yolo_detector = yolo.load_yolov3_detector(self.model_path, self.json_path)
    
    def predict(self, image) -> int:
        """
        Predict the label of the image. +1 if there is a hand, -1 otherwise.
        :param image:      image location or ndarray (BGR)
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
        Each list element is a dictionary, valid keys are:
        "name":                   detected object label
        "percentage_probability": probability for detection
        "box_points":             pixel values of bounding box
                                  [x_min, y_min, x_max, y_max]
        (x_center, y_center, width, height, confidence).
        :param image:      image location or ndarray (BGR)
        :return:           list of bounding boxes
        """
        _, detections = self.yolo_detector.detectObjectsFromImage(input_image=image, output_type="array")
        return detections

class YOLOv8():
    def __init__(self, model_type: str = "small", custom_model = None) -> None:
        if model_type == "small":
            self.model_path = os.path.join(package_directory, 'models', 'yolov8', 'small-yolov8.pt')
        elif model_type == "nano":
            self.model_path = os.path.join(package_directory, 'models', 'yolov8', 'nano-yolov8.pt')
        else:
            raise ValueError('Model type not implemented')
        if custom_model:
            self.model_path = custom_model
        self.yolo_detector = yolo.load_yolov8_detector(self.model_path)
    
    def predict(self, image) -> int:
        """
        Predict the label of the image. +1 if there is a hand, -1 otherwise.
        :param image:      image location or ndarray (BGR)
        :return:           label of the image
        """
        results = self.yolo_detector.predict(source=image, save=False, verbose=False, conf=0.4)
        if len(results[0].boxes) > 0:
            return 1
        else:
            return -1
        
    def predict_boxes(self, image) -> list:
        """
        Return a list of the bounding boxes for detected hands.
        Each list element is a dictionary, valid keys are:
        "name":                   detected object label
        "percentage_probability": probability for detection
        "box_points":             pixel values of bounding box
                                  [x_min, y_min, x_max, y_max]
        (x_center, y_center, width, height, confidence).
        :param image:      image location or ndarray (BGR)
        :return:           list of bounding boxes
        """
        results = self.yolo_detector.predict(source=image, save=False, verbose=False, conf=0.4)[0]
        detections = []
        for box in results.boxes:
            xmin, ymin, xmax, ymax, conf, label_idx  = box.data.cpu().numpy()[0]
            label = self.yolo_detector.names[int(label_idx)]
            box_dict = {"name": label, "percentage_probability": conf,
                        "box_points": [xmin, ymin, xmax, ymax]}
            detections += [box_dict]
        return detections

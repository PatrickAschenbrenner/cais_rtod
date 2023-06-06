from imageai.Detection.Custom import CustomObjectDetection
from ultralytics import YOLO


def load_yolov3_detector(model_path: str,
                         json_path: str) -> CustomObjectDetection:
    """
    Load a trained YOLOv3 model from the given path.
    :param load_path: path to load the model
    :return:          YOLOv3 model
    """
    detector = CustomObjectDetection()
    detector.setModelTypeAsTinyYOLOv3()
    detector.setModelPath(model_path)
    detector.setJsonPath(json_path)
    detector.loadModel()
    return detector


def load_yolov8_detector(model_path: str) -> YOLO:
    """
    Load a trained YOLOv3 model from the given path.
    :param load_path: path to load the model
    :return:          YOLOv3 model
    """
    detector = YOLO(model_path)
    return detector
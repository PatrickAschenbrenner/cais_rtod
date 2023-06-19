import os
import matplotlib.pyplot as plt
from cais_rtod.detector import YOLOv3, YOLOv8


this_directory = os.path.abspath('')
img_hand = os.path.join(this_directory, "data", "hand_example.jpg")

# YOLOv3 example
yolov3_detector = YOLOv3()
boxes = yolov3_detector.predict_boxes(img_hand)
plt.imshow(plt.imread(img_hand))
plt.title("YOLOv3")
for box in boxes:
    x_min, y_min, x_max, y_max = box["box_points"]
    label = box["name"]
    confidence = box["percentage_probability"]
    plt.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], color="red")
    plt.text(x_min, y_min, label + ": " + f"{confidence:.2f}", color="red")
plt.show()

# YOLOv8 example
yolov8_detector = YOLOv8()
boxes = yolov8_detector.predict_boxes(img_hand)
plt.imshow(plt.imread(img_hand))
plt.title("YOLOv8")
for box in boxes:
    x_min, y_min, x_max, y_max = box["box_points"]
    label = box["name"]
    confidence = box["percentage_probability"]
    plt.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], color="red")
    plt.text(x_min, y_min, label + ": " + f"{confidence:.2f}", color="red")
plt.show()

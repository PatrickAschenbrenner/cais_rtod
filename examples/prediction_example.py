import os
from cais_rtod.detector import YOLOv3, YOLOv8, SVM

this_directory = os.path.abspath('')

img_hand = os.path.join(this_directory, "data", "hand_example.jpg")
img_no_hand = os.path.join(this_directory, "data", "no_hand_example.jpg")

# YOLOv3 example
print("YOLOv3:")
yolov3_detector = YOLOv3()
prediction_hand = yolov3_detector.predict(img_hand)
prediction_no_hand = yolov3_detector.predict(img_no_hand)
if prediction_hand == 1:
    print("True positive!")
else:
    print("False negative!")
if prediction_no_hand == -1:
    print("True negative!")
else:
    print("False positive!")
print("")

# YOLOv8 example
print("YOLOv8:")
yolov8_detector = YOLOv8()
prediction_hand = yolov8_detector.predict(img_hand)
prediction_no_hand = yolov8_detector.predict(img_no_hand)
if prediction_hand == 1:
    print("True positive!")
else:
    print("False negative!")
if prediction_no_hand == -1:
    print("True negative!")
else:
    print("False positive!")
print("")

# SVM example
print("SVM:")
svm_detector = SVM()
prediction_hand = svm_detector.predict(img_hand)
prediction_no_hand = svm_detector.predict(img_no_hand)
if prediction_hand == 1:
    print("True positive!")
else:
    print("False negative!")
if prediction_no_hand == -1:
    print("True negative!")
else:
    print("False positive!")
print("")

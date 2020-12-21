# pip install opencv-python

import cv2
import numpy as np

# --------------------------------------------------------------------------
# ref - Pysource on youtube
# code modified by Ranjith V

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# There are 80 classes trained in our files

# names of the classes trained
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# extracting layers from Yolo
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Loading image
img = cv2.imread("sample5.jpg")
# img = cv2.resize(img, None, fx = 0.3, fy = 0.3)
img = cv2.resize(img, None, fx = 3, fy = 3)
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop = False)
net.setInput(blob)
outs = net.forward(output_layers)

# Showing information on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        # Object is considered as detected when confidence score is more than 0.35
        if confidence > 0.35:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle top-left coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # feeding data into the lists
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Non Maximum Suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold = 0.35, nms_threshold = 0.35)

# generating random colors for each class of objects
colors = np.random.uniform(0, 255, size = (len(classes), 3))

# creating object indicators, i.e., frames and labels
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        # frame      (  image,  point1,     pointt2,    color,  thickness)
        cv2.rectangle(  img,    (x, y), (x + w, y + h), color,      2)
        # label    (    image,  text,       org,        fontFace,           fontScale,    color,  thickness)
        cv2.putText(    img,    label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN,   1.323,    (0, 0, 0),     2)

# displaying the image
cv2.imshow("Image", img)
cv2.waitKey(0)

# destroys the windows
cv2.destroyAllWindows()


#THANK YOU
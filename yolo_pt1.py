import numpy as np
import cv2

imgfile = "meja3.jpg"
net = cv2.dnn.readNet("mppi_yolo/yolov3.weights", "mppi_yolo/yolo-tiny.cfg")
with open("mppi_yolo/coco.names", "r") as f:
    classes = f.read().splitlines()
img = cv2.imread(f"mppi_yolo/{imgfile}")
height, width, _ = img.shape

blob = cv2.dnn.blobFromImage(img, 1 / 255, (640, 480), (0, 0, 0), swapRB=True, crop=False)
net.setInput(blob)
output_layers_names = net.getUnconnectedOutLayersNames()
layer_outputs = net.forward(output_layers_names)

boxes = []
confidences = []
class_ids = []

for output in layer_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.7, 0.4)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(boxes), 3))

for i in indexes.flatten():
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    confidence = str(round(confidences[i], 2))
    color = colors[i]
    cv2.rectangle(img,
                  (x, y),
                  (x + w, y + h),
                  color,
                  2)
    cv2.putText(img,
                f"{label} {confidence}",
                (x, y + 20),
                font,
                2,
                (255, 255, 255))
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

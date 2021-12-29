import numpy as np
import cv2

net = cv2.dnn.readNet("mppi_yolo/yolov3.weights",
                      "mppi_yolo/yolov3.cfg")
print("opened weights")
with open("mppi_yolo/coco.names", "r") as f:
    classes = f.read().splitlines()
print("opened classes")

height = 320
width = 640

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
font = cv2.FONT_HERSHEY_PLAIN
color = [255, 255, 255]
print("vision ready")
# ssh -p 1234 hostname
left_boundary = int(width * 3 / 8)
right_boundary = int(width * 5 / 8)
upper_boundary = int(height * 2 / 3)


def vibrate(power):
    pass


def execute_warning(position_vector):
    vibrate(position_vector[0] * 100)
    vibrate(position_vector[1] * 100)
    vibrate(position_vector[2] * 100)


def position_warning(x_pos, y_pos):
    # pos_mat : [upper, left, right]
    pos_mat = [0, 0, 0]
    if y_pos > upper_boundary:
        pos_mat[0] = 1
    if x_pos < left_boundary:
        pos_mat[1] = 1
    elif x_pos > right_boundary:
        pos_mat[2] = 1
    execute_warning(pos_mat)
    return pos_mat


def main():
    while True:
        _, img = cap.read()

        blob = cv2.dnn.blobFromImage(img, np.divide(1, 255), (width, height), (0, 0, 0), swapRB=True, crop=False)
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
                    center_x = int(np.multiply(detection[0], width))
                    center_y = int(np.multiply(detection[1], height))

                    w = int(np.multiply(detection[2], width))
                    h = int(np.multiply(detection[3], height))

                    x = int(np.subtract(center_x, np.divide(w, 2)))
                    y = int(np.subtract(center_y, np.divide(h, 2)))

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.4)

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                center_x = x + w // 2
                center_y = y + h // 2
                label = str(classes[class_ids[i]])
                confidence = str(np.round_(confidences[i], 2))
                pos_mat = position_warning(center_x, center_y)
                # cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.circle(img, (center_x, center_y), 3, color, 2)
                cv2.putText(img, f"{label} {confidence}", (x, y + 20), font, 2, color)

        cv2.imshow("image", img)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


main()

import cv2
import matplotlib.pyplot as plt
from picamera.array import PiRGBArray as pi_rgb
from picamera import PiCamera as picam

confidence_threshold = 0.45  # Threshold to detect object
font = cv2.FONT_HERSHEY_COMPLEX
color = [255, 255, 255]
height = 320
width = 640

class PiCam:
    def __init__(self):
        self.cam = picam()
        self.cam.resolution = (width, height)
        self.cam.framerate = 10
        self.raw_cap = pi_rgb(picam, size=self.cam.resolution)
        time.sleep(0.1)


def focal_length(measured_distance, real_width, width_in_rf_image):
    foc_length = (width_in_rf_image * measured_distance) / real_width
    return foc_length


def distance_finder(foc_len, real_face_width, face_width_in_frame):
    distance = (real_face_width * foc_len) / face_width_in_frame
    return distance


camera = PiCam()
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

for frame in camera.cam.capture_continuous(camera.raw_cap, format="bgr", use_video_port=True):
    img = frame.array
    class_ids, confidences, boundary_boxes = net.detect(img, confThreshold=confidence_threshold)

    if len(class_ids) != 0:
        for classId, confidence, box in zip(class_ids.flatten(), confidences.flatten(), boundary_boxes):
            cv2.rectangle(img, box, color=color, thickness=2)
            cv2.putText(img,
                        classNames[classId - 1].upper(),
                        (box[0] + 10, box[1] + 30),
                        font, 1, color, 2)
            cv2.putText(img,
                        str(round(confidence * 100, 2)),
                        (box[0] + 200, box[1] + 30),
                        font, 1, color, 2)

    cv2.imshow("IEE3061 IoT", img)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

import cv2
from scipy.ndimage.filters import gaussian_filter, convolve
from mppi_frame import Frame
import time

vid = cv2.VideoCapture(0)
canny_min = 100
canny_max = 120
sigma_val = 2
while vid.isOpened():
    ret, frame = vid.read()
    if ret:
        current_frame = Frame(frame)
        cv2.imshow("frame", current_frame.canny(canny_min, canny_max, sigma_val))
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
vid.release()
cv2.destroyAllWindows()

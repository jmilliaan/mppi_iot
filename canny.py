import cv2
from scipy.ndimage.filters import gaussian_filter, convolve
from mppi_frame import Frame
import time

vid = cv2.VideoCapture(0)
canny_min = 100
canny_max = 120
sigma_val = 1
while vid.isOpened():
    ret, frame = vid.read()
    if ret:
        current_frame = Frame(frame)
        canny_frame = current_frame.canny(canny_min, canny_max, sigma_val)
        # sum = current_frame.bw - canny_frame
        cv2.imshow("frame", current_frame.blur(sigma_val))
        cv2.imshow("canny frame", canny_frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
vid.release()
cv2.destroyAllWindows()

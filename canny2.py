import cv2
from mppi_frame import Frame
import time

vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
x_dim = 640
y_dim = 480
vid.set(3, x_dim)
vid.set(4, y_dim)
canny_min = 80
canny_max = 120
sigma_val = 1
x_count = 0
y_count = 0
ret, frame = vid.read()
current_frame = Frame(frame)
canny_frame = current_frame.canny(canny_min,
                                  canny_max,
                                  sigma_val)
t_res, thresh = cv2.threshold(canny_frame, 127, 255, 0)
vid.release()
# sum = current_frame.bw - canny_frame
print("Capture Done")
badlist = current_frame.reverse_knn_joy(thresh, 20)
print("Badlist Done")

for i in badlist:
    try:
        thresh[i[1]][i[0]] = 0
    except IndexError:
        print("meng")
        pass

print("Filter Done")
while True:
    # cv2.imshow("frame", current_frame.reverse_knn_joy(3))
    x_loc = x_count % x_dim
    y_loc = y_count % y_dim
    # cv2.circle(thresh,
    #            (x_loc, y_loc),
    #            1,
    #            (255, 255, 255),
    #            -1)
    cv2.imshow("canny", canny_frame)
    cv2.imshow("thresh", thresh)
    key = cv2.waitKey(1)
    x_count += 1
    if x_loc == 0:
        y_count += 1
    if key == ord('q'):
        break

cv2.destroyAllWindows()

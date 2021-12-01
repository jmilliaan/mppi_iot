import cv2
from scipy.ndimage.filters import gaussian_filter, convolve


class Frame:
    def __init__(self, frame):
        self.raw_frame = frame
        self.bw = cv2.cvtColor(self.raw_frame, cv2.COLOR_BGR2GRAY)

    def blur(self, sigma_val):
        blurred = gaussian_filter(self.bw, sigma=sigma_val)
        return blurred

    def canny(self, minval, maxval, sigma_val):
        cannydetector = cv2.Canny(self.blur(sigma_val), minval, maxval)
        return cannydetector

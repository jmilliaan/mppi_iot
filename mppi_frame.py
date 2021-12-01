import cv2
from scipy.ndimage.filters import gaussian_filter, convolve


class Frame:
    def __init__(self, frame):
        self.raw_frame = frame
        self.bw = cv2.cvtColor(self.raw_frame, cv2.COLOR_BGR2GRAY)
        self.canny_list = [[]]

    def blur(self, sigma_val):
        blurred = gaussian_filter(self.bw, sigma=sigma_val)
        return blurred

    def canny(self, minval, maxval, sigma_val):
        cannydetector = cv2.Canny(self.blur(sigma_val), minval, maxval)
        return cannydetector

    def get_area_magnitude(self, bigmat, xloc, yloc, dim):
        size = int(dim * 2 + 1)
        x_loc_relative = xloc - dim
        y_loc_relative = yloc - dim

        mag = 0
        try:
            for x in range(size):
                for y in range(size):
                    mag += not bigmat[y + y_loc_relative - dim][x + x_loc_relative - dim]
        except IndexError:
            pass
        return mag

    def reverse_knn_joy(self, img, dim):
        bad_list = []
        x_size = len(img[0])
        y_size = len(img)
        empty = (dim * 2 + 1) ** 2
        full_threshold = empty // 20
        for x in range(dim, x_size - dim):
            for y in range(dim, y_size - dim):
                current_mag = empty - self.get_area_magnitude(img, x, y, dim)
                if current_mag >= full_threshold:
                    bad_list.append((x, y))
        return bad_list

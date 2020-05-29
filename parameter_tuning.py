import os
from helper_functions import *

class EdgeFinder:
    def __init__(self, image, threshold1=0, threshold2=0, rho=0, theta=0, \
                 hough_threshold=0, min_line_len=0, max_line_gap=0, kernel_size=1, \
                 x_offset=0, y_offset=0):
        self.image = image
        self._threshold1 = threshold1
        self._threshold2 = threshold2
        self._rho = rho
        self._theta = theta
        self._hough_threshold = hough_threshold
        self._min_line_len = min_line_len
        self._max_line_gap = max_line_gap
        self._kernel_size = kernel_size
        self._roi_x_offset = x_offset
        self._roi_y_offset = y_offset

        self.create_window()

    def create_window(self):

        def onchangeThreshold1(pos):
            self._threshold1 = pos
            self._render()

        def onchangeThreshold2(pos):
            self._threshold2 = pos
            self._render()
        
        def onchangeRho(pos):
            self._rho = pos
            self._render()
        
        def onchangeTheta(pos):
            self._theta = pos
            self._render()
            
        def onchangeHoughThreshold(pos):
            self._hough_threshold = pos
            self._render()
            
        def onchangeMinLineLen(pos):
            self._min_line_len = pos
            self._render()
            
        def onchangeMaxLineLen(pos):
            self._max_line_gap = pos
            self._render()
            
        def onchangeKernelSize(pos):
            self._kernel_size = pos
            self._kernel_size += (self._kernel_size + 1) % 2  # make sure the filter size is odd
            self._render()

        def onchangeROIX(pos):
            self._roi_x_offset = pos
            self._render()
        
        def onchangeROIY(pos):
            self._roi_y_offset = pos
            self._render()

        cv2.namedWindow("edges")

        cv2.createTrackbar('threshold1', 'edges', self._threshold1, 500, onchangeThreshold1)
        cv2.createTrackbar('threshold2', 'edges', self._threshold2, 500, onchangeThreshold2)
        cv2.createTrackbar('rho', 'edges', self._rho, 20, onchangeRho)
        cv2.createTrackbar('theta', 'edges', self._theta, 45, onchangeTheta)
        cv2.createTrackbar('hough_threshold', 'edges', self._hough_threshold, 100, onchangeHoughThreshold)
        cv2.createTrackbar('min_line_len', 'edges', self._min_line_len, 100, onchangeMinLineLen)
        cv2.createTrackbar('max_line_gap', 'edges', self._max_line_gap, 100, onchangeMaxLineLen)
        cv2.createTrackbar('kernel_size', 'edges', self._kernel_size, 21, onchangeKernelSize)
        cv2.createTrackbar('roi_x_offset', 'edges', self._roi_x_offset, 100, onchangeROIX)
        cv2.createTrackbar('roi_y_offset', 'edges', self._roi_y_offset, 100, onchangeROIY)
        self._render()

        cv2.waitKey(0)
        cv2.destroyWindow("edges")


    def set_image(self, image):
        self.image = image
        self._render()

    def edgeImage(self):
        return self._edge_img

    def smoothedImage(self):
        return self._smoothed_img

    def _render(self):
        
        ysize, xsize = self.image.shape[0], self.image.shape[1]
        gray_img = grayscale(self.image)
        gauss_img = gaussian_blur(gray_img, self._kernel_size)
        canny_img = canny(gauss_img, self._threshold1, self._threshold2)
        vertices =  np.array([[(xsize//2+self._roi_x_offset, ysize//2+self._roi_y_offset), \
                                (xsize//2-self._roi_x_offset, ysize//2+self._roi_y_offset), \
                                (0, ysize), (xsize, ysize)]], dtype=np.int32)
        canny_img = region_of_interest(canny_img, vertices)
        line_img = hough_lines(canny_img, self._rho, self._theta*np.pi/180, self._hough_threshold, \
                               self._min_line_len, self._max_line_gap)

        cv2.polylines(line_img, [vertices], True, (0, 0, 255), 2)

        
        final_img = cv2.cvtColor(weighted_img(line_img, initial_img), cv2.COLOR_RGB2BGR)
        

        cv2.imshow('edges',  final_img)
        print(self._threshold1, self._threshold2, self._rho, self._theta, self._hough_threshold, \
                self._min_line_len, self._max_line_gap, self._kernel_size, \
                self._roi_x_offset, self._roi_y_offset)


if __name__ == '__main__':
    edge_finder = None
    for ff in os.listdir("test_images/"):
        print(ff)
        initial_img = mpimg.imread("test_images/" + ff)
        if not edge_finder:
            edge_finder = EdgeFinder(initial_img, threshold1=100, threshold2=250, rho=1, theta=2, \
                    hough_threshold=25, min_line_len=20, max_line_gap=40, kernel_size=7, \
                    x_offset=20, y_offset=0)
        else:
            edge_finder.set_image(initial_img)
            edge_finder.create_window()
     

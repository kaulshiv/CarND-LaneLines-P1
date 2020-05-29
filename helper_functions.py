import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=4):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    slopes = []
    midpoints = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            slopes.append((y2-y1)/(x2-x1))
            midpoints.append(((x1+x2)/2, (y1+y2)/2))
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
#     positive_slopes = [slope for slope in slopes if slope > 0]
#     positive_midpoints = [midpoints[idx] for idx, slope in enumerate(slopes) if slope > 0]
#     negative_slopes = [slope for slope in slopes if slope < 0]
#     negative_midpoints = [midpoints[idx] for idx, slope in enumerate(slopes) if slope < 0]
    
#     right_slope = np.median(positive_slopes)
#     right_mid_x, right_mid_y = np.median([x for x, y in positive_midpoints]), np.median([y for x, y in positive_midpoints])
#     left_slope = np.median(negative_slopes)
#     left_mid_x, left_mid_y = np.median([x for x, y in negative_midpoints]), np.median([y for x, y in negative_midpoints])
                                         
                                         
#     ry1, ry2 = img.shape[0], img.shape[0]//2 + 50
#     rx1, rx2 = int((ry1-right_mid_y)//right_slope + right_mid_x ), int((ry2-right_mid_y)//right_slope + right_mid_x) 
#     ly1, ly2 = img.shape[0], img.shape[0]//2 + 50
#     lx1, lx2 = int((ly1-left_mid_y)//left_slope + left_mid_x), int((ly2-left_mid_y)//left_slope + left_mid_x)
# #     print((rx1, ry1), (rx2, ry2))
# #     print((lx1, ly1), (lx2, ly2))
#     print(positive_slopes)
#     cv2.line(img, (rx1, ry1), (rx2, ry2), color, thickness)
#     cv2.line(img, (lx1, ly1), (lx2, ly2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


import cv2
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import sklearn as sk

from moviepy.editor import VideoFileClip

test_images = 'problem_images/'
test_videos = 'test_videos/'

test_images_output = 'test_images_output/'
test_videos_output = 'test_videos_output/'

clip_name = None
clip_seq = 0

# In order to overcome issues caused by shadows and marks on the road, we will filter image by white and yellow
# using HSV colorspace. The filter range is given by filter_ranges that contains three ranges. The first is for
# pure white, the second is for near white which can be any color with low saturation, and the third for yellow
hsv_ranges = [(np.array([0, 0, 120], dtype = "uint8"), np.array([0, 0, 255], dtype = "uint8")),
                (np.array([0, 0, 220], dtype = "uint8"), np.array([180, 20, 255], dtype = "uint8")), 
                (np.array([18, 80, 120], dtype = "uint8"), np.array([25, 255, 255], dtype = "uint8"))]

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

fits = [None, None]
fit_thresh = 75
def classify_line(line, median):
    '''
    Simple classification for the lines. 
    Return -1 for lines on the left, 1 on the right, 0 if the side cannot be determined, 
           < -1 or > 1 if filtered out as noizy with the returned value the distance, the sign indicate it is on left (< -1) or on the right (> -1)

    Parameters:
    line: the line
    median: the median of the image

    When in a video sequence, we use the previous detection to filter out noisy lines:
    1. both end point of a line need to either be on the right or left of middle of the lane from previous detection
    2. both end point must be in a distance threshoild, fit_thresh, for the line to be accepted
    When not in a video sequence, we take the median to decide whether a line is on the left or right lane
    '''
    if fits[0] != None and fits[1] != None:
        x1_min, x1_max = fits[0][1](line[1]), fits[1][1](line[1])
        x2_min, x2_max = fits[0][1](line[3]), fits[1][1](line[3])
        if line[0] <= (x1_min+x1_max)/2 and line[2] <= (x2_min+x2_max)/2:
            if abs(x1_min-line[0]) < fit_thresh and abs(x2_min-line[2]) < fit_thresh:
                return -1
            else:
                return -max(abs(x1_min-line[0]), abs(x2_min-line[2]))
        elif line[0] >= (x1_min+x1_max)/2 and line[2] >= (x2_min+x2_max)/2:
            if abs(x1_max-line[0]) < fit_thresh and abs(x2_max-line[2]) < fit_thresh:
                return 1
            else:
                return max(bs(x1_max-line[0]), abs(x2_max-line[2]))
        else:
            return 0
    else:
        return -1 if (line[0] < median[0] and line[2] < median[0]) else 1 if (line[0] > median[0] and line[2] > median[0]) else 0

def fitline(lines, previous=None):
    ''' 
    Fit the lines with a first order polynominal
    Return: (ymin, f) where ymin is the minimal y coordinate of the lines, f is the polynominal function

    Parameters:
    lines: the lines
    previous: result of the previous fit
    '''
    if len(lines) == 0:
        return previous
    x = [a[0] for a in lines] + [a[2] for a in lines]
    y = [a[1] for a in lines] + [a[3] for a in lines]

    # weight points by their line length, penaltize relatively short lines
    ylen = [abs(a[1] - a[3]) for a in lines]
    min = np.min(ylen)
    max = np.max(ylen)
    if max - min < 1:
        w = None
    else:
        w = np.exp((ylen - min)/(max - min) + 1.0)
        w = np.repeat(w, 2) # each weight value is for two end points

    z = np.polyfit(y[:], x, 1, w=w) # weighted linear polynominal fit of the points, y will be altered if not copied
    f = np.poly1d(z)
    return np.min(y), f

def interpolate(left, right, bottom):
    ''' 
    Interpolate the lines by fitting the end points with a linear polynomial function.
    bottom: specify bottom of the image, the line will extend to the bottom of the image
    the same as bottom so there is no extension.
    '''
    fits[0] = fitline(left, fits[0])
    fits[1] = fitline(right, fits[1])
    lines = []
    if fits[0]:
        lines = [(int(fits[0][1](fits[0][0])), fits[0][0], int(fits[0][1](bottom)), bottom)]
    if fits[1]:
        lines += [((int(fits[1][1](fits[1][0])), fits[1][0], int(fits[1][1](bottom)), bottom))]
    return lines

def draw_lines(img, lines, color=[255, 0, 0, 1], thickness=2):
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
    if lines is None:
        return
    # filter the lines to exclude lines that are nearly horizontal
    slop_threshold = math.cos(0.15 * np.pi) / math.sin(0.15 * np.pi)
    filtered_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if math.fabs(y2 - y1) > 0:
                m = (x2 - x1) / (y2 - y1) 
                if math.fabs(m) < slop_threshold:
                    filtered_lines += [[x1, y1, x2, y2]]
    # compute the left and right lanes according to the lines' slops
    left = []
    right = []
    filtered_left = []
    filtered_right = []
    for line in filtered_lines:
        c = classify_line(line, [int(img.shape[1] * 0.5), img.shape[0]])
        if c == -1:
            left += [line]
        elif c == 1:
            right += [line]
        elif c < -1:
            filtered_left += [line[0], line[1], line[2], line[3], -c]
        elif c > 1:
            filtered_right += [line[0], line[1], line[2], line[3], c]

    # Handle rare situation where all lines are filtered out
    if len(left) == 0 and len(filtered_left) > 0:
        left = filtered_left.sort(key=lambda x:x[4])[0:3]
    if len(right) == 0 and len(filtered_right) > 0:
        right = filtered_right.sort(key=lambda x:x[4])[0:3]
    
    # interpolate lines on the left(right) for the left(right) boundary of the lane
    lines = interpolate(left, right, img.shape[0])
    
    # Draw the lines
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    if lines is not None:
        draw_lines(line_img, lines, thickness=8)
        return line_img
    else:
        return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def rgb2hsv(img):
    """Applies the RGB to HSV transform"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

def filter_hsv(img):
    ''' 
    Filter the white and yellow colors of the RGB image by HSV 
    '''
    hsv = rgb2hsv(img)
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    
    for (lower, upper) in hsv_ranges:
        # filter range
        mask |= cv2.inRange(hsv, lower, upper)
    # Bitwise-AND mask and original image
    gray = grayscale(img)
    return cv2.bitwise_and(gray, gray, mask= mask)

def find_lane(image, fig = None):
    ''' 
    Find lane in the image
    Fig: when given, images produced during the course of the processaing will be displayed
    '''
    global clip_name, clip_seq
    width = image.shape[1] - 1
    height = image.shape[0] - 1
    gray_img = filter_hsv(image)
    if fig:
        fig.add_subplot(1,4,1)
        plt.imshow(gray_img, cmap='gray')
        
    # Convert the image to gray scale first to simplify processing
    # = grayscale(filtered_img)
    gray_img = gaussian_blur(gray_img, 5)
    
    # Use canny filter to find edges
    gray_img = canny(gray_img, 40, 120)
    if fig:
        fig.add_subplot(1,4,2)
        plt.imshow(gray_img, cmap='gray')

    pts = np.array([[60, height], [width * 0.4, height * 0.65], [width * 0.6, height * 0.65], [width - 60, height]], np.int32)
    gray_img = region_of_interest(gray_img, [pts])
    if fig:
        fig.add_subplot(1,4,3)
        plt.imshow(gray_img, cmap='gray')
    
    # connect lines using hough transform
    lane_img = hough_lines(gray_img, 1, np.pi/180, 15, 40, 150)
    if fig:
        fig.add_subplot(1,4,4)
        plt.imshow(lane_img)

    if clip_name:
        mpimg.imsave(test_videos_output + "{0}{1}-hough.jpg".format(clip_name, clip_seq), lane_img)
    
    #put everything together
    return weighted_img(lane_img, image)

if not os.path.exists(test_videos_output):
    os.makedirs(test_videos_output)

if not os.path.exists(test_images_output):
    os.makedirs(test_images_output)

for image in os.listdir(test_images):
    if image.endswith(".jpg") or image.endswith(".jpeg") or image.endswith(".png"):
        fig = plt.figure(figsize=(12, 48), dpi=80, facecolor='w', edgecolor='k')
        img = mpimg.imread(test_images + image)
        if img.shape[2] > 3:
            img = img[:,:,0:3]
        image_name = image
        lane_img = find_lane(img, fig)
        fig = plt.figure(figsize=(12, 24), dpi=80, facecolor='w', edgecolor='k')
        fig.add_subplot(1,1,1)
        plt.imshow(lane_img)
        mpimg.imsave(test_images_output + image, lane_img)

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    global clip_name, clip_seq
    if clip_name:
        mpimg.imsave(test_videos_output + "{0}{1}.jpg".format(clip_name, clip_seq), image)

    img = find_lane(image)

    if clip_name:
        mpimg.imsave(test_videos_output + "{0}{1}-detect.jpg".format(clip_name, clip_seq), img)
        
    clip_seq += 1
    return img

fits = [None, None]
white_output = test_videos_output + 'solidWhiteRight.mp4'
#clip_name = "solidWhiteRight"
clip_seq = 0
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

fits = [None, None]
yellow_output = test_videos_output + 'solidYellowLeft.mp4'
#clip_name = "solidYellowLeft"
clip_seq = 0
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)

fits = [None, None]
challenge_output = 'test_videos_output/challenge.mp4'
#clip_name = "challenge"
clip_seq = 0
clip2 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip2.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)
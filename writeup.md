# Overview
The goals / steps of this project are the following:

Make a pipeline that finds lane lines on the road
Reflect on your work in a written report

# The Approach
In order to avoid issues caused by shadows, marks and unexpected objects, HSV colorspace for lane detection. In most scenarios, lanes on road or highways are marked using white and yellow colors. By filtering the hue portion of the HSV colorspace for white and yellow regions, the algorithm works much better with shadow conditions. The lane detection pipeline works as follows:

## The Pipeline
1. Convert image from RGB colorspace to HSV colorspace using cv2.cvtColor()
2. Filter the converted HSV image with white and yellow color using cv2.inRanges() with white and yellow HSV color ranges. This results in a boolean mask that can be used to extract the white and yellow regions of the image
3. Mask the original image with the resulting mask
4. Filter the image using gaussian filter of a kernal size 5
5. Perform Canny filter to extract the contour
6. Mask off the resulting contour image with ROI
7. Perform Hough transformation to yield a set of lane lines
8. Class the lines for left and right lanes
9. Fit the end points of lines belong to each of the left and right side of lane with a first degree polynomial function using cv.polyfit() function. This results in two functions of x = a + by.
10. Draw the lanes with the resulting polynomial functions, starting from the bottom of the image to the top end point detected.

# Line classification
The following line classification approaches have been experimented:

## Classified using line slope
This approach classifies left lines and right lines according to their slops:
> m = (x2 - x1) / (y2 - y1)
> The line is classified as left if m < 0, right otherwise

This algorithm classifies the lines with respect to the vertical line slop. It does not work well with lanes that go sideway, or 
nose lines that can go any directions.

In stead of using the vertical line for classification, will it be better to use the average slop. This turned out not to be a good idea, as the average is strongly affected by the number of lines identitied and will therefore be **biased** and cannnot be used for the classification purpose.

## Classified with respect to the position of the line
This approach classifies left lines and right lines according to their x coordinate with respected to the expected lane center:
> cx = image.shape[0] / 2
> left if x1 < cx and x2 < cx, right if x1 > cx and x2 > cx, abandoned otherwise

This approach work will is the lane goes approaimately vertically upward, otherwise some good lines will be skipped.


## Classified using One Class SVM
One class SVM (Support Vector Machine) fit a set of unclassified points, and perdict (classified) a given point into a binary value representing the left and the right (whether -1 or 1 is related to the right does not matter).
> from sklearn import svm
> P = no.array(lines).reshape(len(lines) * 2, 2)
> fit = svm.OneClassSvm(P)
> for the end points, p1, p2 of a line the line is right or left if fit.perdict(p1) and fit.perdict(p1) has the same value.

This approach works to some certain extends, but is not robust enough to be considered as a good approach.

# Issues Encountered And Resolutions
There are few issues encountered during the course of the project that are worth describing here:
## Bad Lane Detection
The origin detection algorithm did not work well with lighting condition, marks or pigement on roads. Conditions like this caused extra contours from Canny filter, and results in lanes that are way off.
### Solution
The solution used, as described above, is to use HSV to extract white and yellow regions of the road to reduce the issues.
### Further Improvements
1. The current implement generates a mask from the HSV image, and use the mask to mask off the unwanted regions in the original image. However, the HSV color ranges used for the filter is constant. It might be better to adjust the intensity of filtering ranges with the average intensity in the ROI, or to normalize the image's intensity before applying the filter.
2. Some image enhancement might also improve the detection, this is a possible area for future improvement
## Classification of Lines
Current implementation of line classification for the left and right sides of lanes simply compares the x coordinates with the middle of the images. This will not work for lanes that are not straight.
### Further Improvements
Further work should try to use a better classification algorithm, like SVM (Support Vector Machine)
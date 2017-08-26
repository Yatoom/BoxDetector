import imutils
import cv2

import helpers
from shape_detector import ShapeDetector
from helpers import show_image
import numpy as np

# the shapes can be approximated better
image = cv2.imread("more_images/19.jpg")
resized = imutils.resize(image, width=400)

hsv_image = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_image)
# show_image("Hue", h)
# show_image("Saturation", s)
# show_image("Value", v)


display_image = imutils.resize(image, width=400)
ratio = display_image.shape[0] / float(resized.shape[0])
back_resized = imutils.resize(image, width=400)
# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(resized, (3, 3), 0)
cannied = imutils.auto_canny(v)
dilation = cv2.dilate(cannied, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=5)
closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
erosion = cv2.erode(closing, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=3)

# show_image("Input image", display_image)
# show_image("Grayscale image", gray)
# show_image("Blurred image", blurred)
show_image("Cannied image", cannied)
show_image("Closed image", erosion)

# thresh = cv2.threshold(blurred, avg_gray_color, 255, cv2.THRESH_BINARY)[1]

# find contours in the thresholded image and initialize the shape detector
contours = cv2.findContours(erosion.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = contours[0] if imutils.is_cv2() else contours[1]

# Find biggest contour
biggest = None
biggest_size = 0
for contour in contours:
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    new_size = cv2.contourArea(box)

    # Check if contour does not start in top left
    M = cv2.moments(contour)
    if M["m00"] == 0:
        continue

    # helpers.show_contour(contour, display_image, ratio=ratio)
    # helpers.show_box(contour, display_image, ratio=ratio)

    # Update if size is bigger
    if new_size > biggest_size:
        biggest_size = new_size
        biggest = contour

helpers.show_box(biggest, display_image, ratio=ratio)
cv2.waitKey(0)

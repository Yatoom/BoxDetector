import imutils
import cv2
from shape_detector import ShapeDetector
from helpers import show_image
import numpy as np

# the shapes can be approximated better
image = cv2.imread("images/1.png")
resized = imutils.resize(image, width=100)
ratio = image.shape[0] / float(resized.shape[0])
back_resized = imutils.resize(image, width=100)
# gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(resized, (3, 3), 0)
cannied = imutils.auto_canny(resized)
# cannied = cv2.Canny(gray, 100, 200)
_, cnts, _ = cv2.findContours(cannied, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

show_image("Input image", image)
# show_image("Grayscale image", gray)
# show_image("Blurred image", blurred)
show_image("Cannied image", cannied)

# thresh = cv2.threshold(blurred, avg_gray_color, 255, cv2.THRESH_BINARY)[1]

# find contours in the thresholded image and initialize the shape detector
contours = cv2.findContours(cannied.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = contours[0] if imutils.is_cv2() else contours[1]
sd = ShapeDetector()

# loop over the contours
for contour in contours:
    print("contour")
    # compute the center of the contour, then detect the name of the
    # shape using only the contour
    M = cv2.moments(contour)

    if M["m00"] == 0:
        continue

    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    shape = sd.detect(contour)

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = box * ratio
    box = np.int0(box)

    # multiply the contour (x, y)-coordinates by the resize ratio,
    # then draw the contours and the name of the shape on the image
    contour = contour.astype("float")
    contour *= ratio
    contour = contour.astype("int")
    cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
    rect = cv2.boundingRect(contour)



    cv2.putText(image, "{} ({})".format(shape, cv2.contourArea(contour)), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)

    # show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)

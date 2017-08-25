import imutils
import cv2
from shape_detector import ShapeDetector
from helpers import show_image
import numpy as np


# the shapes can be approximated better
image = cv2.imread("images/1.png")
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])
b, g, r = cv2.split(resized)

# show_image("blue", b)
# show_image("green", g)
# show_image("red", r)

avg_red = np.average(r)
avg_green = np.average(g)
avg_blue = np.average(b)

thresh_red = cv2.threshold(r, avg_red + 10, 255, cv2.THRESH_BINARY)[1]
thresh_green = cv2.threshold(g, avg_green + 10, 255, cv2.THRESH_BINARY)[1]
thresh_blue = cv2.threshold(b, avg_blue + 10, 255, cv2.THRESH_BINARY)[1]

show_image("blue", thresh_blue)
show_image("green", thresh_green)
show_image("red", thresh_red)

for item in [thresh_green, thresh_blue, thresh_red]:
    contours = cv2.findContours(item.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        contour = contour.astype("float")
        contour *= ratio
        contour = contour.astype("int")
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)

        # show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(0)

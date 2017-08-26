import cv2
import imutils
import numpy as np

for i in range(1, 40):
    input = cv2.imread("more_images/{}.jpg".format(i))
    image = imutils.resize(input, width=400)

    result = cv2.GaussianBlur(image, (1, 1), 0)
    result = imutils.auto_canny(result)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("result", result)
    cv2.waitKey(0)

    contours = cv2.findContours(result.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if imutils.is_cv2() else contours[1]

    # loop over the contours
    biggest_size = 0
    biggest = None

    for c in contours:
        cv2.drawContours(image, [c], -1, (0, 0, 255), 4)

        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 4)

        # Draw box around the contour
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], -1, (255, 0, 0), 2)

        # Calculate size
        new_size = cv2.contourArea(box)
        if new_size > biggest_size:
            biggest_size = new_size
            biggest = box

    # Draw the biggest box
    cv2.drawContours(image, [biggest], -1, (255, 255, 255), 2)

    cv2.imshow("image", image)
    cv2.waitKey(0)
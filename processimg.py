#!/usr/bin/env python3

import numpy as np
import cv2

def disp_scaled(name, image, delay=0):
    cv2.imshow(name, cv2.resize(image, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA))
    cv2.waitKey(0)



fil = "sample.png"
img = cv2.imread(fil)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY)
#thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 191, 2)
kernel = np.ones((6,6), np.uint8)
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#disp_scaled("morph", closed)

contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cutoffs = (1000, 5000)
out = []
for c in contours:
    if cutoffs[0] < cv2.contourArea(c) < cutoffs[1]:
        out.append(c)
contours = np.asarray(out)

cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

disp_scaled("original", img)
#disp_scaled("threshold", thresh)



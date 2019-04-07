#!/usr/bin/env python3
import numpy as np
import cv2
from PIL import Image
import io
import os


def disp(name, image, delay=0):
    cv2.imshow(name, image)
    cv2.waitKey(delay)


fil = "sample.png"
img = cv2.imread(fil)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY)
kernel = np.ones((6, 6), np.uint8)
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cutoffs = (1000, 5000)
out = []
moments = []
for c in contours:
    if cutoffs[0] < cv2.contourArea(c) < cutoffs[1]:
        out.append(c)
        moments.append(cv2.moments(c))

contours = np.asarray(out)
cv2.drawContours(img, contours, -1, (0, 255, 0), 5)

masked = np.zeros_like(img)
cv2.drawContours(img, contours, -1, (0, 255, 0), 5)

disp("original", img)

#!/usr/bin/env python3

import numpy as np
import cv2

def disp_scaled(image):
    cv2.imshow("Current Output", cv2.resize(image, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA))
    cv2.waitKey(0)



fil = "sample.png"
img = cv2.imread(fil)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY)
#thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 191, 2)

disp_scaled(img)
disp_scaled(thresh)


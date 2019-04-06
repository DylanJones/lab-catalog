import cv2
import numpy as np

vid = cv2.VideoCapture("filtered.avi")

stitcher: cv2.Stitcher = cv2.Stitcher_create()

frames = [vid.read() for i in range(10)]

status, pano = stitcher.stitch(frames)

print(status)

cv2.imshow("pano", pano)
cv2.waitKey(0)
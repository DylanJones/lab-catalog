#!/usr/bin/env python3

import numpy as np
import cv2
from PIL import Image
import io
import os
import json
from google.cloud import vision
from google.cloud.vision import types

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/home/alexander/Downloads/lab-catalog.json"

def disp_scaled(name, image, delay=0):
    cv2.imshow(name, cv2.resize(image, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA))
    cv2.waitKey(0)



fil = "pano_manual_full.png"
img = cv2.imread(fil)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
#thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 191, 2)
kernel = np.ones((6,6), np.uint8)
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#disp_scaled("morph", closed)



contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cutoffs = (1000, 5000)
out = []
moments = []
for c in contours:

    if cutoffs[0] < cv2.contourArea(c) < cutoffs[1]:
        out.append(c)
        moments.append(cv2.moments(c))

contours = np.asarray(out)



############################### DESIGNED FOR ROTATED IMAGES; CORRECT LATER -> OR NOT
size = (160, 100)
hsize = (80, 50)
coords = {}
cropable = Image.open(fil)

################## INITIALIZE DA GOOGLE
client = vision.ImageAnnotatorClient()
file_name = os.path.join(os.path.dirname(__file__), 'text1.png')

print(f"Generating text for ~{len(contours)} labels...")
for i in range(len(contours)):
    if i % 100 == 0:
        print(i)
    mts = moments[i]
    cx = int(mts['m10']/mts['m00'])
    cy = int(mts['m01']/mts['m00'])
    cropable.crop((cx - hsize[0], cy - hsize[1], cx + hsize[0], cy + hsize[1])).save("text1.png")
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()
    image_google = types.Image(content=content)
    response = client.text_detection(image=image_google)
    text = response.text_annotations
    if len(text) <= 0:
        response = client.document_text_detection(image=image_google)
        text = response.text_annotations
        if len(text) <= 0:
            continue
    text = text[0].description
    text = text.strip().upper().replace(' ', '')
    width = cropable.width
    height = cropable.height
    coords[text] = {'x1': (cx - hsize[0]) / width, 'y1': (cy - hsize[1]) / height, 'x2': (cx + hsize[0]) / width, 'y2': (cy + hsize[1]) / height}

with open("coordinate_data.json", "w") as jfile:
    jfile.write(json.dumps(coords))


cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

disp_scaled("original", img)
cv2.imwrite("contouredBoi.png",img)
#disp_scaled("threshold", thresh)



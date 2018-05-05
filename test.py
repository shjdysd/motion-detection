import cv2
import numpy as np
from scipy import misc
import os
import sys
from numpy import *
from matplotlib import pyplot as plt
import Canny
from Scene import *

if os.path.isdir("./res") == False:
    os.mkdir("./res")

# preprocessing video, reduce effect from camera shaking
cap = cv2.VideoCapture('./videos/hz.mp4')
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
accumulate = np.zeros([360,640])
count = 0
while(cap.isOpened()):
    ret, img = cap.read()
    if ret == True:
        count += 1
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float64)
        accumulate += img
    else:
        cap.release()
accumulate /= count
canny = Canny.Canny(accumulate)
kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
canny = cv2.filter2D(canny, -1, kernel)
misc.imsave('./res/canny.jpg', canny)


cap = cv2.VideoCapture('./videos/hz.mp4')
if (cap.isOpened() == False):
    print("Error opening video stream or file")

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
videoWriter = cv2.VideoWriter('./res/output.avi', fourcc, 24.0, (640, 360))

ret, img = cap.read()
count = 0

scene = Scene(img, canny, 'HS')

while(cap.isOpened()):
    ret, img = cap.read()
    if ret == True:
        result = scene.update_scene(img)
        misc.imsave('./res/res' + str(count) + '.jpg', result)
        frame = cv2.imread('./res/res' + str(count) + '.jpg')
        videoWriter.write(frame)
        count += 1
    else:
        break

# When everything done, release the video capture object
cap.release()
videoWriter.release()

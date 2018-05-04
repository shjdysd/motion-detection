import cv2
import numpy as np
from scipy import misc
import os
import sys
from numpy import *
from matplotlib import pyplot as plt
import time
import LK
from Functions import *
import HS

if os.path.isdir("./res") == False:
    os.mkdir("./res")

cap = cv2.VideoCapture('./videos/hz.mp4')
if (cap.isOpened() == False):
    print("Error opening video stream or file")

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
videoWriter = cv2.VideoWriter('./res/output.avi', fourcc, 24.0, (640, 360))

ret, old = cap.read()
old = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
prvs = old.astype(np.float64)
count = 0

#model = LK.LK()
model = HS.HS()

while(cap.isOpened()):
    ret, new = cap.read()
    if ret == True:
        curr = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
        curr = curr.astype(np.float64)

        #op_flow = model.lucas_kanade_np(curr, old)
        op_flow = model.HornSchunck(curr, prvs)

        result = np.abs(curr - prvs)
        result -= np.median(result)    
        result = drawRectangle(result, new, op_flow, np.max(result) * 0.1)
        misc.imsave('./res/res' + str(count) + '.jpg', result)
        frame = cv2.imread('./res/res' + str(count) + '.jpg')
        videoWriter.write(frame)
        prvs = curr
        count += 1
    else:
        break

# When everything done, release the video capture object
cap.release()
videoWriter.release()

import cv2
import numpy as np
from scipy import misc
import os, sys
from numpy import *
from matplotlib import pyplot as plt
import time
import LK

if os.path.isdir("./res") == False:
    os.mkdir("./res")

cap = cv2.VideoCapture('./hz.mp4')
if (cap.isOpened()== False): 
    print("Error opening video stream or file")

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
videoWriter = cv2.VideoWriter('output.avi',fourcc, 24.0, (640, 360))
 
ret, old = cap.read()
old = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
count = 0
model = LK.LK()
while(cap.isOpened()):
    ret, new = cap.read()
    if ret == True:
        new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
        """
        result = model.settleFrame(new)
        misc.imsave('./res/res' + str(count) + '.jpg', result)
        frame = cv2.imread('./res/res'+ str(count) +'.jpg')
        """
        result = model.lucas_kanade_np(new, old)
        misc.imsave('./res/res' + str(count) + '.jpg', result)
        frame = cv2.imread('./res/res'+ str(count) +'.jpg')
        #cv2.waitKey(100) 
        videoWriter.write(frame)
        old = new
        count += 1
    else:
        break

# When everything done, release the video capture object
cap.release()
videoWriter.release() 

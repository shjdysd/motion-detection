import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
import os, sys
from numpy import *
from matplotlib import pyplot as plt
import time
import LK

if os.path.isdir("./res") == False:
    os.mkdir("./res")

cap = cv2.VideoCapture('./test.mp4')
if (cap.isOpened()== False): 
    print("Error opening video stream or file")

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
videoWriter = cv2.VideoWriter('output.avi',fourcc, 24.0, (640,360))
 
ret, old = cap.read()
old = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
count = 0
model = LK.LK()
while(cap.isOpened()):
    ret, new = cap.read()
    if ret == True:
        new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
        
        result = model.settleFrame(new)
        misc.imsave('./res/res' + str(count) + '.jpg', result)
        frame = cv2.imread('./res/res'+ str(count) +'.jpg')
        '''
        op_flow = np.abs(model.lucas_kanade_np(new, old))
        old = new
        result = (op_flow[...,0] **2 + op_flow[...,1]**2)**0.5   
        misc.imsave('./res/res' + str(count) + '.jpg', result)
        frame = cv2.imread('./res/res'+ str(count) +'.jpg')
        '''
        #cv2.waitKey(100) 
        videoWriter.write(frame)
        count += 1
    else:
        break

# When everything done, release the video capture object
cap.release()
videoWriter.release() 

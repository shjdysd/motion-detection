import cv2
from scipy import misc
import os
from Canny import *
from Scene import *
from visualize import *

if os.path.isdir("./res") == False:
    os.mkdir("./res")

videoAddress = './videos/2.mp4'

canny = preprocessing(videoAddress)

cap = cv2.VideoCapture(videoAddress)
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
        writeCountToTxt(count)
        videoWriter.write(frame)
        count += 1
    else:
        break
# When everything done, release the video capture object
cap.release()
videoWriter.release()
#visualization()

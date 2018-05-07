##########################################################################################
#
# Desc: implementation of Canny edge detection and preprocessing the video
# 
# P.S. cv2.GaussianBlur() and cv2.filter2D() are used due to the efficiency and 
#      could be replaced by self-implemented function in comments behaind
#
###########################################################################################
from scipy import misc
import numpy as np
import cv2
from Functions import * 

def directive(image):
    Ix = np.zeros(image.shape)
    Iy = np.zeros(image.shape)
    Ix[0:-1, 1:-1] = (image[0:-1, 2:] - image[0:-1, :-2])/2
    Iy[1:-1, 1:-1] = (image[2:, 1:-1] - image[:-2, 1:-1])/2
    return Ix, Iy

def magnitude(x, y):
    newImage = np.zeros(x.shape)
    newImage = (x**2 + y**2)**0.5
    return newImage

def getDirection(Ix, Iy):
    direction = np.zeros(Ix.shape)
    for i in range(0, len(Ix)):
        for j in range(0, len(Ix[0])):
            if Ix[i, j] == 0:
                if Iy[i, j] == 0:
                    angle = 0
                elif Iy[i, j] > 0:
                    angle = 90
                else:
                    angle = 270
            else:
                angle = np.arctan(Iy[i, j] / Ix[i, j]) / np.pi * 180
                if Ix[i, j] > 0 and Iy[i, j] < 0:
                    angle += 360
                if Ix[i, j] < 0:
                    angle += 180
            direction[i, j] = angle
    return direction

def mapping(img):
    newImage = np.zeros_like(img)
    maxValue = np.max(img)
    newImage = img / maxValue * 255
    return newImage
    
def noneMax(image, direction):
    newImage = np.zeros(image.shape)
    row = image.shape[0]
    col = image.shape[1]
    for i in range(0, row):
        for j in range(0, col):
            val = image[i, j]
            d = direction[i, j]
            if d > 180: d -= 180
            if d <= 22.5 or d > 157.5:
                if i > 0 and val < image[i-1, j]:
                    image[i, j] = 0
                elif i < row - 1 and val < image[i+1, j]:
                    image[i, j] = 0
            elif d <= 67.5:
                if j > 0 and i > 0 and val < image[i-1, j-1]:
                    image[i, j] = 0
                elif j < col - 1 and i < row - 1 and val < image[i+1, j+1]:
                    image[i, j] = 0
            elif d <= 112.5:
                if j > 0 and val < image[i, j-1]:
                    image[i, j] = 0
                elif j < col - 1 and val < image[i, j+1]:
                    image[i, j] = 0
            else:
                if j > 0 and i < row - 1 and val < image[i+1, j-1]:
                    image[i, j] = 0
                elif j < col - 1 and i > 0 and val < image[i-1, j+1]:
                    image[i, j] = 0          
    return image
   
def dualThreshold(img):
    DT = np.zeros(img.shape)           
    TL = 0.05 * np.max(img)
    TH = 0.4 * np.max(img)
    for i in range(1, len(DT)-1):
        for j in range(1, len(DT[0])-1):
            if (img[i, j] < TL):
                DT[i, j] = 0
            elif (img[i, j] > TH):
                DT[i, j] = 1
            elif ((img[i-1, j-1:j+1] < TH).any() or (img[i+1, j-1:j+1]).any() 
                  or (img[i, [j-1, j+1]] < TH).any()):
                DT[i, j] = 1
    return DT * 255

def Canny(img):
    img = cv2.GaussianBlur(img, (5,5), 0)     # img = GuassianBlur(img)
    img = img.astype(np.float64)
    Iy, Ix = directive(img)
    mag = magnitude(Ix, Iy)
    direction = getDirection(Ix, Iy)
    mag = mapping(mag)
    mag = noneMax(mag, direction)
    mag = dualThreshold(mag)
    return mag

def preprocessing(videoAddress):
    cap = cv2.VideoCapture(videoAddress)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    accumulate = np.zeros([360, 640])
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
    canny = Canny(accumulate)
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    canny = cv2.filter2D(canny, -1, kernel)
    misc.imsave('./res/canny.jpg', canny)
    return canny

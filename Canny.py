from scipy import misc
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import cv2
from sklearn.cluster import MiniBatchKMeans
import DisjointSet
import queue

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
            if Ix[i][j] == 0:
                if Iy[i][j] == 0:
                    angle = 0
                elif Iy[i][j] > 0:
                    angle = 90
                else:
                    angle = 270
            else:
                angle = np.arctan(Iy[i][j]/Ix[i][j]) / np.pi * 180
                if Ix[i][j] > 0 and Iy[i][j] < 0:
                    angle += 360
                if Ix[i][j] < 0:
                    angle += 180
            direction[i][j] = angle
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
            val = image[i][j]
            d = direction[i][j]
            if d > 180: d -= 180
            if d <= 22.5 or d > 157.5:
                if i > 0 and val < image[i-1][j]:
                    image[i][j] = 0
                elif i < row - 1 and val < image[i+1][j]:
                    image[i][j] = 0
            elif d <= 67.5:
                if j > 0 and i > 0 and val < image[i-1][j-1]:
                    image[i][j] = 0
                elif j < col - 1 and i < row - 1 and val < image[i+1][j+1]:
                    image[i][j] = 0
            elif d <= 112.5:
                if j > 0 and val < image[i][j-1]:
                    image[i][j] = 0
                elif j < col - 1 and val < image[i][j+1]:
                    image[i][j] = 0
            else:
                if j > 0 and i < row - 1 and val < image[i+1][j-1]:
                    image[i][j] = 0
                elif j < col - 1 and i > 0 and val < image[i-1][j+1]:
                    image[i][j] = 0          
    return image
   
def dualThreshold(NMS):
    DT = np.zeros(NMS.shape)           
    TL = 0.05 * np.max(NMS)
    TH = 0.4 * np.max(NMS)
    for i in range(1, len(DT)-1):
        for j in range(1, len(DT[0])-1):
            if (NMS[i, j] < TL):
                DT[i, j] = 0
            elif (NMS[i, j] > TH):
                DT[i, j] = 1
            elif ((NMS[i-1, j-1:j+1] < TH).any() or (NMS[i+1, j-1:j+1]).any() 
                  or (NMS[i, [j-1, j+1]] < TH).any()):
                DT[i, j] = 1
    return DT * 255

def Canny(img):
    img = cv2.GaussianBlur(img, (5,5), 0)
    img = img.astype(np.float64)
    Iy, Ix = directive(img)
    mag = magnitude(Ix, Iy)
    direction = getDirection(Ix, Iy)
    mag = mapping(mag)
    mag = noneMax(mag, direction)
    mag = dualThreshold(mag)
    return mag

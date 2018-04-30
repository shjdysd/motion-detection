from scipy import misc
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import cv2

def directive(image):
    Ix = np.zeros(image.shape)
    Iy = np.zeros(image.shape)
    Ix[0:-1, 1:-1] = (image[0:-1, 2:] - image[0:-1, :-2])/2
    Iy[1:-1, 1:-1] = (image[2:, 1:-1] - image[:-2, 1:-1])/2
    return Ix, Iy

def smooth(image, filterKernel):
    k = len(filterKernel)
    #in x direction
    newImage = np.zeros(image.shape)
    for i in range(0, len(image)):
        for j in range(0, len(image[0])):
            if j < k//2 or len(image[0]) - j - 1 < k//2:
                newImage[i, j] = 0
            else:
                count = 0
                for n in range(0, k):
                    count += image[i][j+k//2-n] * filterKernel[n]
                newImage[i][j] = count
    #in y direction
    image = newImage
    for j in range(0, len(image[0])):
        for i in range(0, len(image)):
            if i < k//2 or len(image) - i - 1 < k//2: 
                newImage[i][j] = 0
            else:
                count = 0
                for n in range(0, k):
                    count += image[i+k//2-n][j] * filterKernel[n]
                newImage[i][j] = count
    return newImage

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
    minValue = np.min(img)
    newImage = (img - minValue) / (maxValue - minValue) * 255
    return newImage

def noneMax(image, direction, min):
    newImage = np.zeros(image.shape)
    row = image.shape[0]
    col = image.shape[1]
    for i in range(0, row):
        for j in range(0, col):
            val = image[i][j]
            if val < min: 
                image[i][j] = 0
                continue
            d = direction[i][j]
            if d > 180: d -= 180
            if d >= 22.5 and d > 157.5:
                if i > 0 and val < image[i-1][j]:
                    image[i][j] = 0
                if i < row - 1 and val < image[i+1][j]:
                    image[i][j] = 0
            elif d <= 67.5:
                if j > 0 and i > 0 and val < image[i-1][j-1]:
                    image[i][j] = 0
                if j < col - 1 and i < row - 1 and val < image[i+1][j+1]:
                    image[i][j] = 0
            elif d <= 112.5:
                if j > 0 and i < row - 1 and val < image[i+1][j-1]:
                    image[i][j] = 0
                if j < col - 1 and i > 0 and val < image[i-1][j+1]:
                    image[i][j] = 0
            else:
                if j > 0 and val < image[i][j-1]:
                    image[i][j] = 0
                if j < col - 1 and val < image[i][j+1]:
                    image[i][j] = 0
    return image

def Canny(img, min, max):
    img = img.astype(np.int32)
    Ix, Iy = directive(img)
    mag = magnitude(Ix, Iy)
    direction = getDirection(Ix, Iy)
    mag = mapping(mag)
    mag = noneMax(mag, direction, min)
    return mag
   

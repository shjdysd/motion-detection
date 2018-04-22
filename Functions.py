from scipy import misc
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os, sys

def convertToGray(image):
    rowNum = image.shape[0]
    colNum = image.shape[1]
    grayImage = np.zeros((rowNum, colNum))
    for i in range(0, rowNum):
        for j in range(0, colNum):
            grayImage[i][j] = average(image[i][j])
    return grayImage

def average(pixel):
    if type(pixel) is np.uint8:
        return float(pixel)
    return float(pixel[0])/3 + float(pixel[1])/3 + float(pixel[2])/3
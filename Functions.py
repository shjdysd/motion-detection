from scipy import misc
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2


def directive(image):
    Ix = np.zeros(image.shape)
    Iy = np.zeros(image.shape)
    Ix[0:-1, 1:-1] = (image[0:-1, 2:] - image[0:-1, :-2]) / 2
    Iy[1:-1, 1:-1] = (image[2:, 1:-1] - image[:-2, 1:-1]) / 2
    return Ix, Iy


def smooth(image, filterKernel):
    k = len(filterKernel)
    # in x direction
    newImage = np.zeros(image.shape)
    for i in range(0, len(image)):
        for j in range(0, len(image[0])):
            if j < k // 2 or len(image[0]) - j - 1 < k // 2:
                newImage[i, j] = 0
            else:
                count = 0
                for n in range(0, k):
                    count += image[i][j + k // 2 - n] * filterKernel[n]
                newImage[i][j] = count
    # in y direction
    image = newImage
    for j in range(0, len(image[0])):
        for i in range(0, len(image)):
            if i < k // 2 or len(image) - i - 1 < k // 2:
                newImage[i][j] = 0
            else:
                count = 0
                for n in range(0, k):
                    count += image[i + k // 2 - n][j] * filterKernel[n]
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
                angle = np.arctan(Iy[i][j] / Ix[i][j]) / np.pi * 180
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
            if d > 180:
                d -= 180
            if d <= 22.5 or d > 157.5:
                if i > 0 and val < image[i - 1][j]:
                    image[i][j] = 0
                elif i < row - 1 and val < image[i + 1][j]:
                    image[i][j] = 0
            elif d <= 67.5:
                if j > 0 and i > 0 and val < image[i - 1][j - 1]:
                    image[i][j] = 0
                elif j < col - 1 and i < row - 1 and val < image[i + 1][j + 1]:
                    image[i][j] = 0
            elif d <= 112.5:
                if j > 0 and val < image[i][j - 1]:
                    image[i][j] = 0
                elif j < col - 1 and val < image[i][j + 1]:
                    image[i][j] = 0
            else:
                if j > 0 and i < row - 1 and val < image[i + 1][j - 1]:
                    image[i][j] = 0
                elif j < col - 1 and i > 0 and val < image[i - 1][j + 1]:
                    image[i][j] = 0
    return image


def dualThreshold(NMS):
    DT = np.zeros(NMS.shape)
    # 定义高低阈值
    TL = 0.05 * np.max(NMS)
    TH = 0.4 * np.max(NMS)
    for i in range(1, len(DT) - 1):
        for j in range(1, len(DT[0]) - 1):
            if (NMS[i, j] < TL):
                DT[i, j] = 0
            elif (NMS[i, j] > TH):
                DT[i, j] = 1
            elif ((NMS[i - 1, j - 1:j + 1] < TH).any() or (NMS[i + 1, j - 1:j + 1]).any()
                  or (NMS[i, [j - 1, j + 1]] < TH).any()):
                DT[i, j] = 1
    return DT * 255


'''
def drawFrame(img, originalImg, threshold):
    isVisited = np.zeros(img.shape)
    row = img.shape[0]
    col = img.shape[1]
    for i in range(row):
        for j in range(col):
            if isVisited[i][j] == 0 and img[i][j] > threshold:
                frame = np.array([i,j,i,j])
                frame= dfs(img, isVisited, i, j, frame, row, col)
                #print((frame[2] - frame[0],frame[3] - frame[1]))
                x = frame[2] - frame[0]
                y = frame[3] - frame[1]
                if x > 9 and y > 9 and x < row/5 and y < col/5:
                    #这里x,y反了
                    cv2.rectangle(originalImg, (frame[1],frame[0]),(frame[3],frame[2]), 255, 1)
    return originalImg
'''


def dfs(img, isVistied, i, j, frame, row, col):
    isVistied[i][j] = 1
    if i < frame[0]:
        frame[0] = i
    elif i > frame[2]:
        frame[2] = i
    if j < frame[1]:
        frame[1] = j
    elif j > frame[3]:
        frame[3] = j
    if i > 0 and isVistied[i - 1][j] == 0 and img[i - 1][j] > 0:
        frame = dfs(img, isVistied, i - 1, j, frame, row, col)
    if i < row - 1 and isVistied[i + 1][j] == 0 and img[i + 1][j] > 0:
        frame = dfs(img, isVistied, i + 1, j, frame, row, col)
    if j > 0 and isVistied[i][j - 1] == 0 and img[i][j - 1] > 0:
        frame = dfs(img, isVistied, i, j - 1, frame, row, col)
    if j < col - 1 and isVistied[i][j + 1] == 0 and img[i][j + 1] > 0:
        frame = dfs(img, isVistied, i, j + 1, frame, row, col)
    if i > 0 and j > 0 and isVistied[i - 1][j - 1] == 0 and img[i - 1][j - 1] > 0:
        frame = dfs(img, isVistied, i - 1, j - 1, frame, row, col)
    if i > 0 and j > col - 1 and isVistied[i - 1][j + 1] == 0 and img[i - 1][j + 1] > 0:
        frame = dfs(img, isVistied, i - 1, j + 1, frame, row, col)
    if i < row - 1 and j > 0 and isVistied[i + 1][j - 1] == 0 and img[i + 1][j - 1] > 0:
        frame = dfs(img, isVistied, i + 1, j - 1, frame, row, col)
    if i < row - 1 and j < col - 1 and isVistied[i + 1][j + 1] == 0 and img[i + 1][j + 1] > 0:
        frame = dfs(img, isVistied, i + 1, j + 1, frame, row, col)
    return frame


def Canny(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = img.astype(np.float64)
    Iy, Ix = directive(img)
    mag = magnitude(Ix, Iy)
    direction = getDirection(Ix, Iy)
    mag = mapping(mag)
    mag = noneMax(mag, direction)
    mag = dualThreshold(mag)
    #mag = drawFrame(mag, img, np.max(mag) * 0.2)
    return mag


def drawFrame(img, originalImg, threshold):
    isVisited = np.zeros(img.shape)
    row = img.shape[0]
    col = img.shape[1]
    rect_list = dict()
    area_list = np.zeros_like(img)
    for i in range(row):
        for j in range(col):
            if isVisited[i][j] == 0 and img[i][j] > threshold:
                frame = np.array([i, j, i, j])
                frame = dfs(img, isVisited, i, j, frame, row, col)
                #print((frame[2] - frame[0],frame[3] - frame[1]))
                x = frame[2] - frame[0]
                y = frame[3] - frame[1]
                if x > 9 and y > 9 and x < row / 5 and y < col / 5:
                    # 这里x,y反了
                    x0 = (frame[3] + frame[1]) // 2
                    y0 = (frame[2] + frame[0]) // 2
                    coordinate = [frame[1], frame[0], frame[3], frame[2]]
                    area = x * y
                    if area_list[y0, x0] < area:
                        area_list[y0, x0] = area
                        rect_list[(x0, y0)] = coordinate

    for i in range(area_list.shape[0]):
        for j in range(area_list.shape[1]):
            area_list, rect_list = rect_suppression(area_list, j, i, rect_list)

    for key in rect_list.keys():
        [x_ul, y_ul, x_br, y_br] = rect_list[key]
        cv2.rectangle(originalImg, (x_ul, y_ul), (x_br, y_br), 255, 1)

    return originalImg


def rect_suppression(areas, x, y, rect_list):
    if areas[y, x] != 0:
        [x_ul, y_ul, x_br, y_br] = rect_list[(x, y)]
        for n in range(y_ul, y_br + 1):
            for m in range(x_ul, x_br + 1):
                if areas[n, m] != 0 and areas[n, m] < areas[y, x]:
                    areas, rect_list = rect_suppression(areas, m, n, rect_list)
                    areas[n, m] = 0
                    rect_list.pop((m, n))
    return areas, rect_list

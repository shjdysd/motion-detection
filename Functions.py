from scipy import misc
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2
from sklearn.cluster import MiniBatchKMeans
import DisjointSet


def k_means(img, originalImg, threshold):
    newImage = np.zeros(img.shape)
    trainSet = []
    row = img.shape[0]
    col = img.shape[1]
    for i in range(row):
        for j in range(col):
            if img[i][j] > threshold:
                trainSet.append([i, j])
    trainSet = np.array(trainSet)
    size = 1
    if trainSet.size > 2000:
        size = 20
    kmeans = MiniBatchKMeans(n_clusters=size, batch_size=100).fit(trainSet)
    center = kmeans.cluster_centers_.astype(np.int32)
    return center


def drawRectangle(img, originalImg, op_flow, threshold):
    pts = k_means(img, originalImg, threshold)
    minDiff = 1000
    DJSet = DisjointSet.DisjointSet()
    for i in range(pts.size // 2):
        for j in range(i + 1, pts.size // 2):
            diff = np.abs(pts[j] - pts[i])
            minDiff = np.minimum(minDiff, diff[0] + diff[1])
    vehicleThreshold = minDiff * 3
    for i in range(pts.size // 2):
        for j in range(pts.size // 2):
            if i == j:
                continue
            diff = np.abs(pts[j] - pts[i])
            if diff[0] + diff[1] < vehicleThreshold:
                DJSet.add(tuple(pts[i]), tuple(pts[j]))
    frame = []
    for leader in DJSet.group:
        if len(DJSet.group[leader]) < 2:
            continue
        frame = [leader[0], leader[1], leader[0], leader[1]]
        for member in DJSet.group[leader]:
            if member[0] < frame[0]:
                frame[0] = member[0]
            elif member[0] > frame[2]:
                frame[2] = member[0]
            if member[1] < frame[1]:
                frame[1] = member[1]
            elif member[1] > frame[3]:
                frame[3] = member[1]
        if frame[3] - frame[1] < 50 or frame[3] - frame[1] > originalImg.shape[1] * 0.5 or frame[2] - frame[0] < 50 or frame[2] - frame[0] > originalImg.shape[0] * 0.5:
            continue
        speed = speedTest(op_flow, frame)
        color = (255, 255, 255)
        if speed < 40:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        cv2.rectangle(originalImg, (frame[1], frame[0]), (frame[3], frame[2]), color, 3)
        cv2.putText(originalImg, str(np.around(speed)), (frame[1] + 5, frame[2] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return originalImg


def speedTest(img, frame):
    x_max = frame[3]
    x_min = frame[1]
    y_max = frame[2]
    y_min = frame[0]

    car = np.array(img[y_min:y_max, x_min:x_max])
    speed = np.zeros(len(car))
    for i in range(len(car)):
        speed[i] = np.max(car[i, :])
    return np.median(speed) / (2.5 * (y_max / img.shape[0]))

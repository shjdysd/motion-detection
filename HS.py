import numpy as np
from scipy import misc
import os, sys
from matplotlib import pyplot as plt
import cv2
import Functions

HS_Kernel =np.array([[1/12, 1/6, 1/12],[1/6, 0, 1/6],[1/12, 1/6, 1/12]], dtype='float64')

minimum = 0
maximum = 100000

Guassian_Kernel = np.array([0.00598,0.060626,0.241843,0.383103,0.241843,0.060626,0.00598])

class HS:

    def __init__(self):
        self = self

    def LinearFilter(self, image, Kernel):
        k = len(Kernel)
        h, w = image.shape
        newImage = np.zeros_like(image)
        for i in range(0, h):
            for j in range(0, w):
                if(image[i, j] != 0):
                    if j < k // 2 or w - j - 1 < k // 2:
                        newImage[i, j] = 0
                    else:
                        count = 0
                        for n in range(0, k):
                            count += image[i][j + k // 2 - n] * Kernel[n]
                        newImage[i][j] = count
        image = newImage
        for j in range(0, w):
            for i in range(0, h):
                if(image[i, j] != 0):
                    if i < k // 2 or h - i - 1 < k // 2: 
                        newImage[i][j] = 0
                    else:
                        count = 0
                        for n in range(0, k):
                            count += image[i + k // 2 - n][j] * Kernel[n]
                        newImage[i][j] = count
        return newImage
        
    def HornSchunck(self, new, old):
        new = cv2.GaussianBlur(new, (5,5), 0)
        old = cv2.GaussianBlur(old, (5,5), 0)
        old = np.array(old, dtype='float64')
        new = np.array(new, dtype='float64')
        assert new.shape == new.shape
        I_x = np.zeros_like(new, dtype='float64')
        I_y = np.zeros_like(new, dtype='float64')
        I_t = np.zeros_like(new, dtype='float64')
        I_x[1:-1, 1:-1] = (new[1:-1, 2:] - new[1:-1, :-2]) / 2.0
        I_y[1:-1, 1:-1] = (new[2:, 1:-1] - new[:-2, 1:-1]) / 2.0
        I_t[1:-1, 1:-1] = new[1:-1, 1:-1] - old[1:-1, 1:-1]
        lambda_c = 10
        repeat_t = 3
        u = np.zeros_like(new, dtype='float64')
        v = np.zeros_like(new, dtype='float64')
        for i in range(repeat_t):
            alpha = np.zeros_like(new)
            _u = cv2.filter2D(u, -1,HS_Kernel);
            _v = cv2.filter2D(v, -1,HS_Kernel);
            alpha = (I_x * _u + I_y * _v + I_t) / (1 + lambda_c * (I_x ** 2 + I_y ** 2)) * lambda_c
            u = _u - alpha * I_x
            v = _v - alpha * I_x
            del alpha, _u, _v

        op_mag = np.sqrt(u ** 2 + v ** 2)
        max_mag = np.max(op_mag)

        return op_mag

    def settleFrame(self, image):
        return Functions.Canny(image, 100, 200)

    def findObjectSet(self, image):
        return null

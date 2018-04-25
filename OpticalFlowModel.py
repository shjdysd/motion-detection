import cv2
import numpy as np
from scipy import misc
import os, sys
from numpy import *
from matplotlib import pyplot as plt
import time

class OpticalFlowModel:

    def __init__(self):
        self = self

    def getOptical(self, new, old):
        windowsSize = 5
        new = cv2.GaussianBlur(new, (5, 5), 1);
        old = cv2.GaussianBlur(old, (5, 5), 1);
        new = cv2.resize(new, (0,0), fx=0.25, fy=0.25) 
        old = cv2.resize(old, (0,0), fx=0.25, fy=0.25) 
        k = windowsSize // 2;
        newImage = new[:]
        It = new - old
        new = cv2.Laplacian(new, cv2.CV_64F)#CV_64F为图像深度
        Ix = cv2.Sobel(new,cv2.CV_64F,1,0,ksize=3)#1，0参数表示在x方向求一阶导数
        Iy = cv2.Sobel(new,cv2.CV_64F,0,1,ksize=3)#0,1参数表示在y方向求一阶导数 
        lamb = 10
        row = newImage.shape[0]
        col = newImage.shape[1]
        u = np.zeros_like(newImage, dtype='float64')
        v = np.zeros_like(newImage, dtype='float64')
        
        for n in range(15):
            for i in range(1, row-1):
                for j in range(1, col-1):
                    _u = (u[i-1, j]+u[i+1, j]+u[i, j-1]+u[i, j+1]) * 0.25
                    _v = (v[i-1, j]+v[i+1, j]+v[i, j-1]+v[i, j+1]) * 0.25
                    alpha = (Ix[i, j] * _u + Iy[i, j] * _v + It[i, j]) / (1 + lamb*(Ix[i, j] ** 2 + Iy[i, j] ** 2)) * lamb
                    u[i, j] = _u - alpha * Ix[i, j]
                    v[i, j] = _v - alpha * Iy[i, j]
                    if (u[i, j] ** 2 + v[i, j] ** 2) ** 0.5 > 1:
                        newImage[i][j] = 0
        '''
        for i in range(k, row - k):
            for j in range(k, col - k):
                A = self.buildA(Ix, Iy, i, j, windowsSize)
                b = self.buildB(It, i, j, windowsSize);
                if np.linalg.det((A.T).dot(A)) != 0:
                    Vpt = np.matrix((A.T).dot(A)).I.dot(A.T).dot(b)
                    if (Vpt[0,0]**2 + Vpt[0,1]**2)**0.5 > 10:
                        newImage[i][j] = 0 
                       ''' 
        return newImage

    def buildA(self, Ix, Iy, x, y, kernelSize):
        #build a kernel containing pixel intensities
        mean = kernelSize // 2
        count = 0
       #home = img[centerX, centerY] #storing the intensity of the center pixel
        A = np.zeros([kernelSize**2, 2])
        for i in range(-mean,mean+1): #advance the x
            for j in range(-mean,mean+1): #advance the y 
                Ax = Ix[x+i][y+j]
                Ay = Iy[x+i][y+j]
                A[count] = np.array([Ax, Ay])
                count += 1
        return A

    def buildB(self, It, x, y, kernelSize):
        mean = kernelSize//2
        count = 0
        B = np.zeros([kernelSize**2])
        for i in range(-mean,mean+1):
            for j in range(-mean,mean+1):
                Bt = It[x+i][y+j]
                B[count] = Bt
                count += 1
        return B
        
if __name__=="__main__":
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    if os.path.isdir("./res") == False:
        os.mkdir("./res")
    cap = cv2.VideoCapture('./test.mp4')
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (160,90))
     
    # Read until video is completed
    ret, old = cap.read()
    old = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
    count = 1
    model = OpticalFlowModel()
    while(cap.isOpened()):
        ret, new = cap.read()   
        if ret == True:
            new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
            opticalImage = model.getOptical(new, old)
            out.write(opticalImage)
            misc.imsave('./res/res' + str(count) + '.bmp', opticalImage)
            count += 1
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    #cv2.destroyAllWindows()

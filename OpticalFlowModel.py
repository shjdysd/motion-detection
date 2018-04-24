import cv2
import numpy as np
from scipy import misc
import os, sys
from numpy import *

class OpticalFlowModel:

    def __init__(self):
        self = self

    def getOptical(self, new, old):
        windowsSize = 5
        k = windowsSize // 2;
        newImage = new[:]
        It = new - old
        new = cv2.Laplacian(new, cv2.CV_64F)#CV_64F为图像深度
        Ix = cv2.Sobel(new,cv2.CV_64F,1,0,ksize=3)#1，0参数表示在x方向求一阶导数
        Iy = cv2.Sobel(new,cv2.CV_64F,0,1,ksize=3)#0,1参数表示在y方向求一阶导数 
        row = newImage.shape[0]
        col = newImage.shape[1]
        for i in range(k, row - k):
            for j in range(k, col - k):
                A = mat(zeros((windowsSize**2, 2)))
                b = mat(zeros((windowsSize**2, 1)))
                for m in range(-k, k+1):
                    for n in range(-k, k+1):
                        A[(m+2)*5+n+2, 0] = Ix[i+m][j+n] 
                        A[(m+2)*5+n+2, 1] = Iy[i+m][j+n]
                        b[(m+2)*5+n+2] = -It[i+m][j+n]
                tensor = A.T * A
                if linalg.det(tensor) != 0:
                    result = (tensor).I * A.T * b
                    val = (result[0]**2 + result[1]**2) * 10
                    if val > 20:
                        newImage[i][j] = 255
        return newImage
        
if __name__=="__main__":
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    if os.path.isdir("./res") == False:
        os.mkdir("./res")
    cap = cv2.VideoCapture('./test.mp4')
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
     
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
            misc.imsave('./res/res' + str(count) + '.png', opticalImage)
            count += 1
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    #cv2.destroyAllWindows()

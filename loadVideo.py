import cv2
import numpy as np
from scipy import misc
import os, sys
from numpy import *

def getOptical(old, new):
	It = new - old
	new = cv2.Laplacian(new, cv2.CV_64F)#CV_64F为图像深度
	Ix = cv2.Sobel(new,cv2.CV_64F,1,0,ksize=5)#1，0参数表示在x方向求一阶导数
	Iy = cv2.Sobel(new,cv2.CV_64F,0,1,ksize=5)#0,1参数表示在y方向求一阶导数	
	cv2.imshow('Ix', Ix)
	"""
	newImage = new[:]
	row = newImage.shape[0]
	col = newImage.shape[1]
	for i in range(1, row - 1):
		for j in range(1, col - 1):
			A = mat(zeros((9, 2)))
			b = mat(zeros((9, 1)))
			for m in range(-1, 2):
				for n in range(-1, 2):
					A[(m+1)*3+n+1, 0] = Ix[i+m][j+n] 
					A[(m+1)*3+n+1, 1] = Iy[i+m][j+n]
					b[(m+1)*3+n+1] = It[i+m][j+n]
			if linalg.det(A.T * A) != 0:
				result = (A.T * A).I * A.T * b
				val = result[0]**2 + result[1]**2
				if val > 10:
					newImage[i][j] = val
					"""
	return Ix
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
i = 1
while(cap.isOpened()):
	ret, new = cap.read()	
	if ret == True:
		new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
		opticalImage = getOptical(old, new)
		misc.imsave('./res/res' + str(i) + '.png', opticalImage)
		i += 1
	else:
		break

# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()

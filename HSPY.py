import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import os

FILTER = 7
count = 0

def HS(im1, im2, alpha, ite,):

    #set up initial velocities
    uInitial = np.zeros([im1.shape[0],im1.shape[1]])
    vInitial = np.zeros([im1.shape[0],im1.shape[1]])

    # Set initial value for the flow vectors
    u = uInitial
    v = vInitial

    # Estimate derivatives
    [fx, fy, ft] = computeDerivatives(im1, im2)

    # Averaging kernel
    kernel=np.matrix([[1/12, 1/6, 1/12],[1/6, 0, 1/6],[1/12, 1/6, 1/12]])

    print(fx[100,100],fy[100,100],ft[100,100])

    # Iteration to reduce error
    for i in range(ite):
        # Compute local averages of the flow vectors
        uAvg = cv2.filter2D(u,-1,kernel)
        vAvg = cv2.filter2D(v,-1,kernel)

        uNumer = (fx.dot(uAvg.T) + fy.dot(vAvg.T) + ft).dot(ft)
        uDenom = alpha + fx**2 + fy**2
        u = uAvg - np.divide(uNumer,uDenom)

        # print np.linalg.norm(u)

        vNumer = (fx.dot(uAvg) + fy.dot(vAvg) + ft).dot(ft)
        vDenom = alpha + fx**2 + fy**2
        v = vAvg - np.divide(vNumer,vDenom)
    return (u,v)

def computeDerivatives(im1, im2):
    # build kernels for calculating derivatives
    kernelX = np.matrix([[-1,1],[-1,1]])*.25 #kernel for computing dx
    kernelY = np.matrix([[-1,-1],[1,1]])*.25 #kernel for computing dy
    kernelT = np.ones([2,2])*.25

    #apply the filter to every pixel using OpenCV's convolution function
    fx = cv2.filter2D(im1,-1,kernelX) + cv2.filter2D(im2,-1,kernelX)
    fy = cv2.filter2D(im1,-1,kernelY) + cv2.filter2D(im2,-1,kernelY)
    # ft = im2 - im1
    ft = cv2.filter2D(im2,-1,kernelT) + cv2.filter2D(im1,-1,-kernelT)
    return (fx,fy,ft)

def smoothImage(img,kernel):
    G = gaussFilter(kernel)
    smoothedImage=cv2.filter2D(img,-1,G)
    smoothedImage=cv2.filter2D(smoothedImage,-1,G.T)
    return smoothedImage

def gaussFilter(segma):
    kSize = 2*(segma*3)
    x = range(-kSize//2,kSize//2,1+1//kSize)
    x = np.array(x)
    G = (1/(2*np.pi)**.5*segma) * np.exp(-x**2/(2*segma**2))
    return G

def compareGraphs():
    plt.ion() #makes it so plots don't block code execution
    plt.imshow(imgNew,cmap = 'gray')
    # plt.scatter(POI[:,0,1],POI[:,0,0])
    for i in range(len(u)):
        if i%5 ==0:
            for j in range(len(u)):
                if j%5 == 0:
                    plt.arrow(j,i,v[i,j]*.00001,u[i,j]*.00001, color = 'red')
                pass
        # print i
    # plt.arrow(POI[:,0,0],POI[:,0,1],0,-5)
    plt.show()

if os.path.isdir("./res") == False:
    os.mkdir("./res")
cap = cv2.VideoCapture('./videoplayback.mp4')
# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (160,90))
 
# Read until video is completed
ret, old = cap.read()
old = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
count = 1
while(cap.isOpened()):
    ret, new = cap.read()   
    new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY) 
    if ret == True:
        imgNew = smoothImage(new,1)
        imgOld = smoothImage(old,1)
        [u,v] = HS(imgOld, imgNew, 1, 100)
        compareGraphs()
        count += 1
        old = new;
    else:
        break

# When everything done, release the video capture object
cap.release()
#cv2.destroyAllWindows()
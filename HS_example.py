from scipy.ndimage.filters import convolve as filter2
import numpy as np
import cv2
import os
from scipy import misc
from matplotlib import pyplot as plt
#
HSKERN =np.array([[1/12, 1/6, 1/12],
                  [1/6,    0, 1/6],
                  [1/12, 1/6, 1/12]],float)

kernelX = np.array([[-1, 1],
                     [-1, 1]]) * .25 #kernel for computing d/dx

kernelY = np.array([[-1,-1],
                     [ 1, 1]]) * .25 #kernel for computing d/dy

kernelT = np.ones((2,2))*.25


def HornSchunck(im1, im2, alpha:float=0.001, Niter:int=8, verbose:bool=False):
    """
    im1: image at t=0
    im2: image at t=1
    alpha: regularization constant
    Niter: number of iteration
    """
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    #set up initial velocities
    uInitial = np.zeros([im1.shape[0],im1.shape[1]])
    vInitial = np.zeros([im1.shape[0],im1.shape[1]])

    # Set initial value for the flow vectors
    U = uInitial
    V = vInitial

    # Estimate derivatives
    [fx, fy, ft] = computeDerivatives(im1, im2)

    if verbose:
        from .plots import plotderiv
        plotderiv(fx,fy,ft)

#    print(fx[100,100],fy[100,100],ft[100,100])

    # Iteration to reduce error
    for _ in range(Niter):
#%% Compute local averages of the flow vectors
        uAvg = filter2(U, HSKERN)
        vAvg = filter2(V, HSKERN)
#%% common part of update step
        der = (fx*uAvg + fy*vAvg + ft) / (alpha**2 + fx**2 + fy**2)
#%% iterative step
        U = uAvg - fx * der
        V = vAvg - fy * der

    return U,V


def computeDerivatives(im1, im2):

    fx = filter2(im1,kernelX) + filter2(im2,kernelX)
    fy = filter2(im1,kernelY) + filter2(im2,kernelY)

   # ft = im2 - im1
    ft = filter2(im1,kernelT) + filter2(im2,-kernelT)

    return fx,fy,ft


if os.path.isdir("./res") == False:
        os.mkdir("./res")
cap = cv2.VideoCapture('./ball.flv')
# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
 
# Read until video is completed
ret, old = cap.read()
old = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
count = 1
while(cap.isOpened()):
    ret, new = cap.read()    
    if ret == True:
        new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
        U, V = HornSchunck(new, old)
        X = np.arange(0, U.shape[1], 1)
        Y = np.arange(0, U.shape[0], 1)
        fig, ax = plt.subplots()
        q = ax.quiver(X, Y, U, V)
        plt.savefig('./res/res' + str(count) + '.png')
        #misc.imsave('./res/res' + str(count) + '.bmp', opticalImage)
        count += 1
        old = new;
    else:
        break

# When everything done, release the video capture object
cap.release()
#cv2.destroyAllWindows()
import numpy as np
from scipy import misc
import os, sys
from matplotlib import pyplot as plt
import cv2
import Functions

HS_Kernel =np.array([1/6, 0, 1/6],float)

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

    def MediumFilter(self, image, k_size=3):
        width = k_size // 2
        kernel = np.zeros([k_size, k_size])
        kernel[:, 1:] = image[:k_size, :k_size - 1]
        result = np.array(image)
        for i in range(width, image.shape[0] - width - 1):
            for j in range(width, image.shape[1] - width - 1):
                next_list =  np.array([image[i - width:i + width + 1, j + width + 1]]).T
                kernel = np.concatenate((kernel[:, 1:], next_list), axis=1)
                sorted_list = kernel.flatten()
                sorted_list.sort()
                result[i, j] = sorted_list[k_size**2 // 2]
                del sorted_list
        return result

    def LineDetection(self, img):
        new = np.zeros_like(img)
        I_x = np.zeros_like(new)
        I_y = np.zeros_like(new)
        I_x[1:-1, 1:-1] = (img[1:-1, 2:] - img[1:-1, :-2]) / 2.0
        I_x[1:-1, 1:-1] = (I_x[1:-1, 2:] - I_x[1:-1, :-2]) / 2.0
        for i in range(1, img.shape[0]-1):
            for j in range(1, img.shape[1]-1):
                if (img[i, j-1] < img[i, j]) and (img[i, j] > img[i, j+1]):
                    new[i, j] = img[i, j]

        I_y[1:-1, 1:-1] = (img[2:, 1:-1] - img[:-2, 1:-1]) / 2.0
        I_y[1:-1, 1:-1] = (I_y[2:, 1:-1] - I_y[:-2, 1:-1]) / 2.0
        for i in range(1, img.shape[0]-1):
            for j in range(1, img.shape[1]-1):
                if (img[i-1, j] < img[i, j]) and (img[i, j] > img[i+1, j]):
                    new[i, j] = img[i, j]

        return new
        
    def HornSchunck(self, new, old):
        #old = self.MediumFilter(np.array(old))
        #new = self.MediumFilter(np.array(new))
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
        repeat_t = 8
        u = np.zeros_like(new, dtype='float64')
        v = np.zeros_like(new, dtype='float64')
        for i in range(repeat_t):
            alpha = np.zeros_like(new)
            _u = self.LinearFilter(u, HS_Kernel);
            _v = self.LinearFilter(v, HS_Kernel);
            alpha = (I_x * _u + I_y * _v + I_t) / (1 + lambda_c * (I_x ** 2 + I_y ** 2)) * lambda_c
            u = _u - alpha * I_x
            v = _v - alpha * I_x
            del alpha, _u, _v

        op_mag = np.sqrt(u ** 2 + v ** 2)
        max_mag = np.max(op_mag)

        u = np.where(op_mag > np.ones(op_mag.shape)*max_mag*0.01, np.ones((u.shape))*255, np.zeros((u.shape)))
        v = np.where(op_mag > np.ones(op_mag.shape)*max_mag*0.01, np.ones((v.shape))*255, np.zeros((v.shape)))

        return np.sqrt(u ** 2 + v ** 2);

    def settleFrame(self, image):
        return Functions.Canny(image, 100, 200)

    def findObjectSet(self, image):
        return null


if __name__=="__main__":
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    if os.path.isdir("./res") == False:
        os.mkdir("./res")
    cap = cv2.VideoCapture('./hz.mp4')
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter('output.avi',fourcc, 24.0, (640,360))
     
    # Read until video is completed
    ret, old = cap.read()
    model = HS()
    old = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
    count = 1
    while(cap.isOpened()):
        ret, new = cap.read()    
        if ret == True:
            new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
            opticalImage = model.HornSchunck(old, new)
            #new = model.Medium_Filter(new)
            misc.imsave('./res/res' + str(count) + '.jpg', opticalImage)
            frame = cv2.imread('./res/res' + str(count) + '.jpg')
            videoWriter.write(frame)
            count += 1
            old = new;
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    #cv2.destroyAllWindows()
    videoWriter.release()


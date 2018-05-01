import numpy as np
from scipy import misc
import os, sys
from matplotlib import pyplot as plt
import cv2

HS_Kernel =np.array([1/6, 0, 1/6],float)

class HS:

    def __init__(self):
        self = self

    def filter(self, image, Kernel):
        k = len(Kernel)
        h, w = image.shape
        newImage = np.zeros_like(image)
        for i in range(0, h):
            for j in range(0, w):
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
                if i < k // 2 or h - i - 1 < k // 2: 
                    newImage[i][j] = 0
                else:
                    count = 0
                    for n in range(0, k):
                        count += image[i + k // 2 - n][j] * Kernel[n]
                    newImage[i][j] = count
        return newImage

        
    def HornSchunck(self, old, new):
        old = np.array(old)
        new = np.array(new)
        assert new.shape == new.shape
        I_x = np.zeros_like(new)
        I_y = np.zeros_like(new)
        I_t = np.zeros_like(new)
        I_x[1:-1, 1:-1] = (new[1:-1, 2:] - new[1:-1, :-2]) / 2.0
        I_y[1:-1, 1:-1] = (new[2:, 1:-1] - new[:-2, 1:-1]) / 2.0
        I_t[1:-1, 1:-1] = new[1:-1, 1:-1] - old[1:-1, 1:-1]
        lambda_c = 10
        repeat_t = 8
        u = np.zeros_like(new)
        v = np.zeros_like(new)
        for i in range(repeat_t):
            alpha = np.zeros_like(new)
            u = self.filter(u, HS_Kernel);
            v = self.filter(v, HS_Kernel);
            alpha = (I_x * u + I_y * v + I_t) / (1 + lambda_c * (I_x ** 2 + I_y ** 2)) * lambda_c
            u = u - alpha * I_x
            v = v - alpha * I_x
            del alpha

        magnitude = np.sqrt(u ** 2 + v ** 2)
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                if (magnitude[i, j] > 100 ) \
                    or ( magnitude[i, j] < -100):
                    new[i, j] = old[i, j]
                else:
                    new[i, j] = 0

        del magnitude, u, v

        return new;

if __name__=="__main__":
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    if os.path.isdir("./res") == False:
        os.mkdir("./res")
    cap = cv2.VideoCapture('./test.mp4')
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter('output.avi',fourcc, 24.0, (640,360))
     
    # Read until video is completed
    ret, old = cap.read()
    old = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
    count = 1
    model = HS()
    while(cap.isOpened()):
        ret, new = cap.read()    
        if ret == True:
            new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
            opticalImage = model.HornSchunck(old, new)
            #out.write(opticalImage)
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





##########################################################################################
#
# Desc: implementation of filters which may be used in this project 
#
###########################################################################################
import numpy as np

Guassian_Kernel = np.array([0.00598,0.060626,0.241843,0.383103,0.241843,0.060626,0.00598])

def MediumFilter(image, k_size=3):
    width = k_size // 2
    kernel = np.zeros([k_size, k_size])
    kernel[:, 1:] = image[:k_size, :k_size - 1]
    result = np.array(image)
    for i in range(width, image.shape[0] - width - 1):
        for j in range(width, image.shape[1] - width - 1):
            next_list = np.array([image[i - width:i + width + 1, j + width + 1]]).T
            kernel = np.concatenate((kernel[:, 1:], next_list), axis=1)
            sorted_list = kernel.flatten()
            sorted_list.sort()
            result[i, j] = sorted_list[k_size**2 // 2]
            del sorted_list
    return result

def LineDetection(img):
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

def LinearFilter(image, Kernel):
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

def filtler2D(image, Kernel):
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
                        for m in range(0, k):
                            count += image[i][j + k // 2 - n] * Kernel[n, m]
                    newImage[i][j] = count
    return newImage

def GuassianBlur(img):
    return filter2D(img, Guassian_Kernel)
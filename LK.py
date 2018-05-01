import cv2
import numpy as np
from scipy import misc
import os, sys
from numpy import *
from matplotlib import pyplot as plt
import time
import pylab
from scipy import signal
import Functions

class LK:

    def __init__(self):
        self = self

    def optical_flow(self, I1g, I2g, window_size, tau=1e-2):
     
        kernel_x = np.array([[-1., 1.], [-1., 1.]])
        kernel_y = np.array([[-1., -1.], [1., 1.]])
        kernel_t = np.array([[1., 1.], [1., 1.]])#*.25
        w = window_size//2 # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
        I1g = I1g / 255. # normalize pixels
        I2g = I2g / 255. # normalize pixels
        # Implement Lucas Kanade
        # for each point, calculate I_x, I_y, I_t
        mode = 'same'
        fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
        fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
        ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode)+signal.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)
        u = np.zeros(I1g.shape)
        v = np.zeros(I1g.shape)
        # within window window_size * window_size
        for i in range(w, I1g.shape[0]-w):
            for j in range(w, I1g.shape[1]-w):
                Ix = fx[i-w:i+w+1, j-w:j+w+1].flatten()
                Iy = fy[i-w:i+w+1, j-w:j+w+1].flatten()
                It = ft[i-w:i+w+1, j-w:j+w+1].flatten()
                Ix2 = Ix * Ix
                Iy2 = Iy * Iy
                Ixy = Ix * Iy
                Ixt = Ix * It
                Iyt = Iy * It
                del Ix, Iy, It
                #b = ... # get b here
                #A = ... # get A here
                # if threshold τ is larger than the smallest eigenvalue of A'A:
        return (u,v)

    def lucas_kanade_np(self, im1, im2, win=15):
        #im1 = cv2.GaussianBlur(im1, (5,5), 0)
        #im2 = cv2.GaussianBlur(im2, (5,5), 0)
        im1 = im1.astype(np.float64)
        im2 = im2.astype(np.float64)
        I_x = np.zeros(im1.shape)
        I_y = np.zeros(im1.shape)
        I_t = np.zeros(im1.shape)
        I_x[1:-1, 1:-1] = (im1[1:-1, 2:] - im1[1:-1, :-2]) / 2 + (im2[1:-1, 2:] - im2[1:-1, :-2]) / 2
        I_y[1:-1, 1:-1] = (im1[2:, 1:-1] - im1[:-2, 1:-1]) / 2 + (im2[2:, 1:-1] - im2[:-2, 1:-1]) / 2
        I_t[1:-1, 1:-1] = im1[1:-1, 1:-1] - im2[1:-1, 1:-1]
        params = np.zeros(im1.shape + (5,)) #Ix2, Iy2, Ixy, Ixt, Iyt
        params[..., 0] = I_x * I_x # I_x2
        params[..., 1] = I_y * I_y # I_y2
        params[..., 2] = I_x * I_y # I_xy
        params[..., 3] = I_x * I_t # I_xt
        params[..., 4] = I_y * I_t # I_yt
        del I_x, I_y, I_t
        cum_params = np.cumsum(np.cumsum(params, axis=0), axis=1)
        del params
        win_params = (cum_params[2 * win + 1:, 2 * win + 1:] - cum_params[2 * win + 1:, :-1 - 2 * win] - cum_params[:-1 - 2 * win, 2 * win + 1:] + cum_params[:-1 - 2 * win, :-1 - 2 * win])
        del cum_params
        op_flow = np.zeros(im1.shape + (2,))
        det = win_params[...,0] * win_params[..., 1] - win_params[..., 2] **2
        op_flow_x = np.where(det != 0, (win_params[..., 1] * win_params[..., 3] - win_params[..., 2] * win_params[..., 4]) / det, 0)
        op_flow_y = np.where(det != 0, (win_params[..., 0] * win_params[..., 4] - win_params[..., 2] * win_params[..., 3]) / det, 0)
        op_flow[win + 1: -1 - win, win + 1: -1 - win, 0] = op_flow_x[:-1, :-1]
        op_flow[win + 1: -1 - win, win + 1: -1 - win, 1] = op_flow_y[:-1, :-1]
        op_flow_x = op_flow[...,0]
        op_flow_y = op_flow[...,1]
        op_mag = (op_flow[...,0]**2+op_flow[...,1]**2)**0.5
        #maxMag = np.max(op_mag)
        #op_flow_x = np.where(op_mag > np.ones(op_mag.shape)*maxMag*0.1, np.ones((op_flow_x.shape))*255, np.zeros((op_flow_x.shape)))
        #op_flow_y = np.where(op_mag > np.ones(op_mag.shape)*maxMag*0.1, np.ones((op_flow_y.shape))*255, np.zeros((op_flow_y.shape)))
        #op_mag = (op_flow_x**2+op_flow_y**2)**0.5
        return op_mag
        #return Functions.Canny(op_mag)

##########################################################################################
#
# Desc: implementation of Lucas–Kanade
#
###########################################################################################
import numpy as np


class LK:

    def __init__(self):
        self = self

    def optical_flow(self, im1, im2, win=15):
        im1 = im1.astype(np.float64)
        im2 = im2.astype(np.float64)
        I_x = np.zeros(im1.shape)
        I_y = np.zeros(im1.shape)
        I_t = np.zeros(im1.shape)
        I_x[1:-1, 1:-1] = (im1[1:-1, 2:] - im1[1:-1, :-2]) / 2 + (im2[1:-1, 2:] - im2[1:-1, :-2]) / 2
        I_y[1:-1, 1:-1] = (im1[2:, 1:-1] - im1[:-2, 1:-1]) / 2 + (im2[2:, 1:-1] - im2[:-2, 1:-1]) / 2
        I_t[1:-1, 1:-1] = im1[1:-1, 1:-1] - im2[1:-1, 1:-1]
        params = np.zeros(im1.shape + (5,)) 
        params[..., 0] = I_x * I_x  
        params[..., 1] = I_y * I_y  
        params[..., 2] = I_x * I_y  
        params[..., 3] = I_x * I_t  
        params[..., 4] = I_y * I_t  
        del I_x, I_y, I_t
        cum_params = np.cumsum(np.cumsum(params, axis=0), axis=1)
        del params
        win_params = (cum_params[2 * win + 1:, 2 * win + 1:] - cum_params[2 * win + 1:, :-1 - 2 * win] - cum_params[:-1 - 2 * win, 2 * win + 1:] + cum_params[:-1 - 2 * win, :-1 - 2 * win])
        del cum_params
        op_flow = np.zeros(im1.shape + (2,))
        det = win_params[..., 0] * win_params[..., 1] - win_params[..., 2] ** 2
        op_flow_x = np.where(det != 0, (win_params[..., 1] * win_params[..., 3] - win_params[..., 2] * win_params[..., 4]) / det, 0)
        op_flow_y = np.where(det != 0, (win_params[..., 0] * win_params[..., 4] - win_params[..., 2] * win_params[..., 3]) / det, 0)
        op_flow[win + 1: -1 - win, win + 1: -1 - win, 0] = op_flow_x[:-1, :-1]
        op_flow[win + 1: -1 - win, win + 1: -1 - win, 1] = op_flow_y[:-1, :-1]
        op_flow_x = op_flow[..., 0]
        op_flow_y = op_flow[..., 1]
        op_mag = (op_flow[..., 0]**2 + op_flow[..., 1]**2)**0.5
        return op_mag

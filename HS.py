#############################################################################################
#
# Desc: implementation of Horn-Schunck
# 
# P.S. cv2.GaussianBlur() and cv2.filter2D() are used due to the efficiency and 
#      could be replaced by self-implemented function in comments behaind
#
##############################################################################################
import numpy as np
import cv2
from Functions import * 

HS_Kernel =np.array([[1/12, 1/6, 1/12],[1/6, 0, 1/6],[1/12, 1/6, 1/12]], dtype='float64')

class HS:

    def __init__(self):
        self = self
        
    def optical_flow(self, new, old):
        new = cv2.GaussianBlur(new, (5,5), 0)         # new = GuassianBlur(new)
        old = cv2.GaussianBlur(old, (5,5), 0)         # old = GuassianBlur(old)
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
        repeat_t = 3
        u = np.zeros_like(new, dtype='float64')
        v = np.zeros_like(new, dtype='float64')
        for i in range(repeat_t):
            alpha = np.zeros_like(new)
            _u = cv2.filter2D(u, -1,HS_Kernel)        # _u = filter2D(u, HS_Kernel)
            _v = cv2.filter2D(v, -1,HS_Kernel)        # _v = filter2D(v, HS_Kernel)
            alpha = (I_x * _u + I_y * _v + I_t) / (1 + lambda_c * (I_x ** 2 + I_y ** 2)) * lambda_c
            u = _u - alpha * I_x
            v = _v - alpha * I_x
            del alpha, _u, _v

        op_mag = np.sqrt(u ** 2 + v ** 2)
        max_mag = np.max(op_mag)

        return op_mag

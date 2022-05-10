#!/usr/bin/env python3

import numpy as np


def thresholding(img):
    thres = np.zeros(img.shape)
    strong = 1.0
    weak = 0.5
    mmax = np.max(img)
    lo, hi = 0.1 * mmax, 0.8 * mmax
    strongs = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            px = img[i][j]
            if px >= hi:
                thres[i][j] = strong
                strongs.append((i, j))
            elif px >= lo:
                thres[i][j] = weak
    return thres, strongs

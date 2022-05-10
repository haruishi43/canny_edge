#!/usr/bin/env python3

import numpy as np
from numpy.fft import fft2, ifft2


def gradient(img):
    # Sobel operator
    op1 = np.array(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ],
    )
    op2 = np.array(
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ],
    )
    kernel1 = np.zeros(img.shape)
    kernel1[:op1.shape[0], :op1.shape[1]] = op1
    kernel1 = fft2(kernel1)

    kernel2 = np.zeros(img.shape)
    kernel2[:op2.shape[0], :op2.shape[1]] = op2
    kernel2 = fft2(kernel2)

    fim = fft2(img)
    Gx = np.real(ifft2(kernel1 * fim)).astype(float)
    Gy = np.real(ifft2(kernel2 * fim)).astype(float)

    G = np.sqrt(Gx**2 + Gy**2)
    Theta = np.arctan2(Gy, Gx) * 180 / np.pi
    return G, Theta

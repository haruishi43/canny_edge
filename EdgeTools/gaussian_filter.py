#!/usr/bin/env python3

import numpy as np
from numpy.fft import fft2, ifft2


def gaussian(img):
    b = np.array(
        [
            [2, 4,  5,  2,  2],
            [4, 9,  12, 9,  4],
            [5, 12, 15, 12, 5],
            [4, 9,  12, 9,  4],
            [2, 4,  5,  4,  2],
        ],
    ) / 156
    kernel = np.zeros(img.shape)
    kernel[:b.shape[0], :b.shape[1]] = b

    fimg = fft2(img)
    fkernel = fft2(kernel)
    fil_img = ifft2(fimg * fkernel)

    return np.abs(fil_img).astype(int)

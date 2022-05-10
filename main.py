#!/usr/bin/env python3

import argparse

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from EdgeTools import (
    edge_detector,
    gaussian,
    gradient,
    nms,
    thresholding,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "image",
        type=str,
        help="path of image",
    )
    args = parser.parse_args()
    return args


def open_image(img_path):
    img = np.array(Image.open(img_path))
    return img


def vis_edge_detector(img):
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Original')

    plt.gray()
    # convert to gray
    img = img[:, :, 0]

    gim = gaussian(img)
    grim, gphase = gradient(gim)
    gmax = nms(grim, gphase)
    thres = thresholding(gmax)
    edge = edge_detector(thres)

    plt.subplot(1, 2, 2)
    plt.imshow(edge)
    plt.axis('off')
    plt.title('Edges')

    plt.show()


def vis_double_thresholding(img):
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Original')

    plt.gray()
    # convert to gray
    img = img[:, :, 0]

    gim = gaussian(img)
    grim, gphase = gradient(gim)
    gmax = nms(grim, gphase)
    thres = thresholding(gmax)

    plt.subplot(1, 2, 2)
    plt.imshow(thres[0])
    plt.axis('off')
    plt.title('Thrs')

    plt.show()


def vis_nms(img):
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Original')

    plt.gray()
    # convert to gray
    img = img[:, :, 0]

    gim = gaussian(img)
    grim, gphase = gradient(gim)
    gmax = nms(grim, gphase)

    plt.subplot(2, 2, 2)
    plt.imshow(gim)
    plt.axis('off')
    plt.title('Gaussian')

    plt.subplot(2, 2, 3)
    plt.imshow(grim)
    plt.axis('off')
    plt.title('Gradient')

    plt.subplot(2, 2, 4)
    plt.imshow(gmax)
    plt.axis('off')
    plt.title('Non-Maximum suppression')

    plt.show()


def vis_gradient(img):

    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Original')

    plt.gray()
    # convert to gray
    img = img[:, :, 0]

    gim = gaussian(img)
    grim, gphase = gradient(gim)

    plt.subplot(2, 2, 2)
    plt.imshow(gim)
    plt.axis('off')
    plt.title('Gaussian')

    plt.subplot(2, 2, 3)
    plt.imshow(grim)
    plt.axis('off')
    plt.title('Gradient')

    plt.show()


def vis_gaussian(img):
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Original')

    plt.gray()
    # convert to gray
    img = img[:, :, 0]

    gim = gaussian(img)

    plt.subplot(1, 2, 2)
    plt.imshow(gim)
    plt.axis('off')
    plt.title('Gaussian')

    plt.show()


if __name__ == '__main__':

    args = parse_args()
    img = open_image(args.image)

    # vis_gaussian(img)
    # vis_gradient(img)
    # vis_double_thresholding(img)
    # vis_nms(img)
    vis_edge_detector(img)

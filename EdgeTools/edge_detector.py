#!/usr/bin/env python3

import numpy as np


def dfs(img, vis, origin, dx, dy):

    def _exists(img, x, y):
        return x >= 0 and x < img.shape[0] and y >= 0 and y < img.shape[1]

    q = [origin]
    while len(q) > 0:
        s = q.pop()
        vis[s] = True
        img[s] = 1
        for k in range(len(dx)):
            for c in range(1, 16):
                nx, ny = s[0] + c * dx[k], s[1] + c * dy[k]
                if (
                    _exists(img, nx, ny)
                    and (img[nx, ny] >= 0.5)
                    and (not vis[nx, ny])
                ):
                    q.append((nx, ny))


def edge_detector(tr):
    img = tr[0]
    strongs = tr[1]

    vis = np.zeros(img.shape, bool)
    dx = [1, 0, -1,  0, -1, -1, 1,  1]
    dy = [0, 1,  0, -1,  1, -1, 1, -1]

    for s in strongs:
        if not vis[s]:
            dfs(img, vis, s, dx, dy)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = 1.0 if vis[i, j] else 0.0

    return img

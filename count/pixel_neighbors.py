# -*- coding: utf-8 -*-
# @Time : 2020/4/9 下午12:25
# @Author : LuoLu
# @FileName: pixel_neighbors.py
# @Software: PyCharm
# @Github ：https://github.com/luolugithub
# @E-mail ：argluolu@gmail.com

import numpy as np
from numpy.lib.stride_tricks import as_strided


def sliding_window(arr, window_size):
    """ Construct a sliding window view of the array"""
    arr = np.asarray(arr)
    window_size = int(window_size)
    if arr.ndim != 2:
        raise ValueError("need 2-D input")
    if not (window_size > 0):
        raise ValueError("need a positive window size")
    shape = (arr.shape[0] - window_size + 1,
             arr.shape[1] - window_size + 1,
             window_size, window_size)
    if shape[0] <= 0:
        shape = (1, shape[1], arr.shape[0], shape[3])
    if shape[1] <= 0:
        shape = (shape[0], 1, shape[2], arr.shape[1])
    strides = (arr.shape[1] * arr.itemsize, arr.itemsize,
               arr.shape[1] * arr.itemsize, arr.itemsize)
    return as_strided(arr, shape=shape, strides=strides)


def cell_neighbors(arr, i, j, d):
    """Return d-th neighbors of cell (i, j)"""
    w = sliding_window(arr, 2 * d + 1)

    ix = np.clip(i - d, 0, w.shape[0] - 1)
    jx = np.clip(j - d, 0, w.shape[1] - 1)

    i0 = max(0, i - d - ix)
    j0 = max(0, j - d - jx)
    i1 = w.shape[2] - max(0, d - i + ix)
    j1 = w.shape[3] - max(0, d - j + jx)

    return w[ix, jx][i0:i1, j0:j1].ravel()


x = np.arange(8 * 8).reshape(8, 8)
print(x)

for d in [1, 2]:
    for p in [(0, 0), (0, 1), (6, 6), (8, 8)]:
        print("-- d=%d, %r" % (d, p))
        print(cell_neighbors(x, p[0], p[1], d=d))

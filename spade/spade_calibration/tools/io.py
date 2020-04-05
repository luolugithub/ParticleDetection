from os import makedirs, getpid
from os.path import exists
from time import localtime, strftime

import numpy as np
from skimage.io import imsave as skimsave


def strtointlist(string):
    return [int(s) for s in string.split(',')]


def strtofloatlist(string):
    return [float(s) for s in string.split(',')]


def load(filename):
    filename = add_npz(filename)
    packed = np.load(filename)
    unpacked = tuple(t[1] for t in sorted(list(packed.items())))
    if len(unpacked) == 1:
        return unpacked[0]
    else:
        return unpacked


def save(filename, obj, compressed=False):
    if compressed:
        npsave = np.savez_compressed
    else:
        npsave = np.savez

    if type(obj) == tuple:
        npsave(filename, *obj)
    else:
        npsave(filename, obj)


def add_npz(filename):
    if not filename.endswith('.npz'):
        filename += '.npz'
    return filename


def load_or_compute(filename, func, inp, recompute=False, compressed=False):
    filename = add_npz(filename)
    if not recompute and exists(filename):
        return load(filename)
    else:
        mkdirs("/".join(filename.split("/")[:-1]))
        res = func(*inp)
        save(filename, res, compressed)
    return res


def mkdirs(*folders):
    for folder in folders:
        makedirs(folder, exist_ok=True)


def imsave(filename, im, labmat, expmask, mask):
    maxi = im.max()
    mini = im.min()
    rgb = np.array((im, im, im))
    prez = np.array((im, im, im))
    for i in (0, 2):
        rgb[i, :, :, :][np.logical_not(mask)] = maxi
        rgb[1, :, :, :][np.logical_not(mask)] = mini
        prez[0, :, :, :][labmat.astype(bool)] = maxi
        prez[1, :, :, :][labmat.astype(bool)] = maxi
        prez[2, :, :, :][labmat.astype(bool)] = mini
    for i in range(1, 4):
        rgb[np.roll(expmask, i, axis=0)] = mini
        rgb[expmask] = maxi
    overlay = (np.concatenate((prez, rgb, np.array((im, im, im))), axis=3)
               .transpose(1, 2, 3, 0) / maxi * 255).astype(np.uint8)
    skimsave(filename, overlay)


def printnow(*args):
    text = ""
    for arg in args:
        text += str(arg)
    now = strftime("%H:%M:%S", localtime())
    pid = '[%s]' % getpid()
    print(pid + now + ' - ' + text)


def imsave_overlay(filename, im, labmat, mask):
    maxi = im.max()
    mini = im.min()
    rgb = np.array((im, im, im))
    for i in (0, 2):
        rgb[0, :, :, :][labmat.astype(bool)] = maxi
        rgb[1, :, :, :][labmat.astype(bool)] = maxi
        rgb[2, :, :, :][labmat.astype(bool)] = mini
        rgb[i, :, :, :][np.logical_not(mask)] = maxi
        rgb[1, :, :, :][np.logical_not(mask)] = mini
    overlay = (np.concatenate((rgb, np.array((im, im, im))), axis=3)
               .transpose(1, 2, 3, 0) / maxi * 255).astype(np.uint8)
    skimsave(filename, overlay)

import numpy as np
from os import listdir


def mip_expertise(expertise):
    corrected_expertise = []
    for obj in expertise:
        corrected_expertise += [(0, obj[1], obj[2])]
    return corrected_expertise


def read_expertise(f):
    ann = []
    with open(f, 'r') as fh:
        dump = fh.readline()
        for line in fh:
            pos = line[:-1].split('\t')
            ann += (int(float(pos[3]) - 1), int(float(pos[2])),
                    int(float(pos[1]))),
    return ann


def read_expertises(expertisesfolder):
    expertisesfiles = [f for f in listdir(expertisesfolder)]
    expertises = {}
    for annfile in expertisesfiles:
        expertises[annfile[:-4]] = read_expertise(expertisesfolder + annfile)
    return expertises


def compare_to_expertise(exp, labmat, return_mask=False):
    tp = 0
    fn = 0
    dd = 0
    expmask = np.zeros(((3,) + labmat.shape), dtype=bool)
    for obj in exp:
        expmask[2, :, :, :][obj] = True
        if labmat[obj]:
            expmask[1, :, :, :][labmat == labmat[obj]] = True
            labmat[labmat == labmat[obj]] = 0
            expmask[2, :, :, :][obj] = False
            tp += 1
        elif expmask[1, :, :, :][obj]:
            expmask[0, :, :, :][obj] = True
            expmask[1, :, :, :][obj] = False
            expmask[2, :, :, :][obj] = False
            dd += 1
        else:
            fn += 1
    expmask[0, :, :, :][labmat.astype(bool)] = True
    fp = len(np.unique(labmat)) - 1
    if return_mask:
        return tp, fp, fn, dd, expmask
    else:
        return tp, fp, fn, dd

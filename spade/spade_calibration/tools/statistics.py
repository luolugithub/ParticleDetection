from skimage.io import imread
from os import listdir, system
import numpy as np
import pandas as pd


def f1score(sens, prec):
    return 2 * sens * prec / (prec + sens)


def prec(tp, fp):
    return tp / (tp + fp)


def sens(tp, gt):
    return tp / gt


def add_scores(df, tp='tp', fp='fp', gt='gt', name='score', precname='prec',
           sensname='sens'):
    df[precname] = prec(df[tp], df[fp])
    df[sensname] = sens(df[tp], df[gt])
    df[name] = f1score(df.sens, df.prec)


def add_scores_and_groundtruths(df, expn, gt='gt'):
    imgs = df.image.unique()
    df.set_index('image', inplace=True)
    for img in imgs:
        df.loc[img, 'ex'] = expn[img]
    add_scores(df, gt=gt)
    df.reset_index(inplace=True)


def best_unique(df, gby, stat):
    return df.loc[df.groupby(gby).agg({stat: pd.Series.argmax})[stat]] \
               .sort_values(by=stat)[::-1]


def best_multiple(df, gby, stat):
    maxval = df.groupby(gby).agg({stat: pd.np.max})
    if type(stat) != list: stat = [stat]
    if type(gby) != list: gby = [gby]
    return df.merge(on=gby + stat, right=maxval.reset_index())


def getimgchar(im):
    immean = round(im.mean())
    imvar = round(im.var())
    immin = im.min()
    immax = im.max()
    imrange = immax - immin
    immedian = round(np.median(im))
    return [immean, immedian, imvar, immin, immax, imrange]


def compileparamstest(paramstestfolder, paramstestfile):
    #  csvhead='img,xymet,cont,thresh,filt,minshape,zmet,zp1,zp2,tp,fp,fn,dd'
    #  system('echo %s > %s' % (csvhead,paramstestfile))
    system('cp /mnt/data/paramstest.csv %s' % paramstestfile)
    system('cd %s; ls | while read filename; do cat $filename; done >> %s'
           % (paramstestfolder, paramstestfile))


#  system('cd %s; ls | while read filename; do cat $filename; done >> %s'
#          % (paramstestfolder[:-2],paramstestfile))
#  with open(paramstestfile,'w') as outfh:
#    outfh.write(csvhead)
#    for f in listdir(paramstestfolder):
#      with open(paramstestfolder+f) as fh:
#        for row in fh:
#          outfh.write(row)

def suffixcolumns(df, suffix, start=1):
    cols = list(df.columns)
    df.columns = cols[:start] + [n + suffix for n in cols[start:]]


def writeimgstats(imgfolder, mes, imgstatsfile, masksfolder, filtersfolder,
                  filtnames):
    with open(imgstatsfile, 'w') as fh:
        fh.write(
            'img,filter,immean,immedian,imvar,immin,immax,imrange,filtsize\n')
        for img in listdir(imgfolder):
            if not img.endswith('.tif'): continue
            im = imread(imgfolder + img)
            maskfile = masksfolder + img
            filterfile = filtersfolder + img
            mask = npload(maskfile)
            filts = loadorcompute(filterfile, genfilters, (im, mask, mes))
            for filti, rfilt in enumerate(filts):
                filt = rfilt[:, 1:1 - (mes * 2), 1:1 - (mes * 2)]
                fh.write(
                    csvstr([img] + [filtnames[filti]] + getimgchar(im[filt]) + \
                           [filt.sum()]))


def writexystats():
    with open(xystatsfile, 'w') as fh:
        fh.write('img,xymet,cont,thresh,filt,minshape,tp,fp,fn,dd\n')
        for xyf in listdir(xyfilesfolder):
            candidates, xylab = npload(xyfilesfolder + xyf)
            met = xyf.split('_')
            xymet = met[0][:-1]
            cont = met[0][-1]
            thresh = met[1][1:]
            minshape = met[2][1:]
            filt = met[3][1:]
            img = '_'.join(met[4:])[:-4]
            tp, fp, fn, dd, expmask = comparetoexpertise(exp[img], xylab)
            fh.write(csvstr(
                [img, xymet, cont, thresh, filt, minshape, tp, fp, fn, dd]))


def writestats(statfile, stats):
    with open(statfile, 'w') as fh:
        fh.write(csvstr(stats))


def csvstr(l, sep=',', term='\n'):
    return sep.join([str(stat) for stat in l]) + term

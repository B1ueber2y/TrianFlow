import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import cv2, skimage
import skimage.io
#import scipy.misc as sm
import imageio as sm


# Adopted from https://github.com/mrharicot/monodepth
def compute_errors(gt, pred, nyu=False):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()

    rmse = (gt - pred)**2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred))**2
    rmse_log = np.sqrt(rmse_log.mean())

    log10 = np.mean(np.abs((np.log10(gt) - np.log10(pred))))

    abs_rel = np.mean(np.abs(gt - pred) / (gt))

    sq_rel = np.mean(((gt - pred)**2) / (gt))

    if nyu:
        return abs_rel, sq_rel, rmse, log10, a1, a2, a3
    else:
        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


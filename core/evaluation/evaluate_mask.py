import os
import numpy as np
import cv2
import functools
import matplotlib.pyplot as plt
import multiprocessing

"""
Adopted from https://github.com/martinkersner/py_img_seg_eval
"""

class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def pixel_accuracy(eval_segm, gt_segm):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i = 0

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i += np.sum(curr_gt_mask)

    if (sum_t_i == 0):
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i

    return pixel_accuracy_


def mean_accuracy(eval_segm, gt_segm):
    '''
    (1/n_cl) sum_i(n_ii/t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    accuracy = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)

        if (t_i != 0):
            accuracy[i] = n_ii / t_i

    mean_accuracy_ = np.mean(accuracy)
    return mean_accuracy_


def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)

    mean_IU_ = np.sum(IU) / n_cl_gt
    return mean_IU_, np.array(IU)


def frequency_weighted_IU(eval_segm, gt_segm):
    '''
    sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    frequency_weighted_IU_ = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        frequency_weighted_IU_[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)

    sum_k_t_k = get_pixel_area(eval_segm)

    frequency_weighted_IU_ = np.sum(frequency_weighted_IU_) / sum_k_t_k
    return frequency_weighted_IU_


'''
Auxiliary functions used during evaluation.
'''


def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask


def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl


def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl


def extract_masks(segm, cl, n_cl):
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")

def read_mask_gt_worker(gt_dataset_dir, idx):
    return cv2.imread(
        gt_dataset_dir + "/obj_map/" + str(idx).zfill(6) + "_10.png", -1)

def load_gt_mask(gt_dataset_dir):
    num_gt = 200
    
    # the dataset dir should be the directory of kitti-2015.
    fun = functools.partial(read_mask_gt_worker, gt_dataset_dir)
    pool = multiprocessing.Pool(5)
    results = pool.imap(fun, range(num_gt), chunksize=10)
    pool.close()
    pool.join()

    gt_masks = []
    for m in results:
        m[m > 0.0] = 1.0
        gt_masks.append(m)
    return gt_masks


def eval_mask(pred_masks, gt_masks, opt):
    grey_cmap = plt.get_cmap("Greys")
    if not os.path.exists(os.path.join(opt.trace, "pred_mask")):
        os.mkdir(os.path.join(opt.trace, "pred_mask"))

    pa_res, ma_res, mIU_res, fwIU_res = 0.0, 0.0, 0.0, 0.0
    IU_res = np.array([0.0, 0.0])

    num_total = len(gt_masks)
    for i in range(num_total):
        gt_mask = gt_masks[i]
        H, W = gt_mask.shape[0:2]

        pred_mask = cv2.resize(
            pred_masks[i], (W, H), interpolation=cv2.INTER_LINEAR)

        pred_mask[pred_mask >= 0.5] = 1.0
        pred_mask[pred_mask < 0.5] = 0.0

        cv2.imwrite(
            os.path.join(opt.trace, "pred_mask",
                         str(i).zfill(6) + "_10_plot.png"),
            grey_cmap(pred_mask))
        cv2.imwrite(
            os.path.join(opt.trace, "pred_mask", str(i).zfill(6) + "_10.png"),
            pred_mask)

        pa_res += pixel_accuracy(pred_mask, gt_mask)
        ma_res += mean_accuracy(pred_mask, gt_mask)

        mIU, IU = mean_IU(pred_mask, gt_mask)
        mIU_res += mIU
        IU_res += IU

        fwIU_res += frequency_weighted_IU(pred_mask, gt_mask)

    return pa_res / 200., ma_res / 200., mIU_res / 200., fwIU_res / 200., IU_res / 200.

from evaluation_utils import *

def process_depth(gt_depth, pred_depth, min_depth, max_depth):
    mask = gt_depth > 0
    pred_depth[pred_depth < min_depth] = min_depth
    pred_depth[pred_depth > max_depth] = max_depth
    gt_depth[gt_depth < min_depth] = min_depth
    gt_depth[gt_depth > max_depth] = max_depth

    return gt_depth, pred_depth, mask


def eval_depth(gt_depths,
               pred_depths,
               min_depth=1e-3,
               max_depth=80, nyu=False):
    num_samples = len(pred_depths)
    rms = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    d1_all = np.zeros(num_samples, np.float32)
    a1 = np.zeros(num_samples, np.float32)
    a2 = np.zeros(num_samples, np.float32)
    a3 = np.zeros(num_samples, np.float32)

    for i in range(num_samples):
        gt_depth = gt_depths[i]
        pred_depth = pred_depths[i]
        mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)
        
        if not nyu:
            gt_height, gt_width = gt_depth.shape
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                            0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
        
        gt_depth = gt_depth[mask]
        pred_depth = pred_depth[mask]
        scale = np.median(gt_depth) / np.median(pred_depth)
        pred_depth *= scale

        gt_depth, pred_depth, mask = process_depth(
            gt_depth, pred_depth, min_depth, max_depth)

        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[
            i] = compute_errors(gt_depth, pred_depth, nyu=nyu)


    return [abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), a1.mean(), a2.mean(), a3.mean()]


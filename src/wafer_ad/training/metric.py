

import numpy as np
from sklearn.metrics import auc
from skimage.measure import label, regionprops

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def eval_seg_pro(
    gt_mask: np.ndarray,
    anomaly_score_map: np.ndarray,
    max_step: int = 800,
    expect_fpr: float = 0.3,
) -> float:
    """
    Optimized PRO AUC evaluation.

    Args:
        gt_mask: (N, H, W) binary ground-truth masks
        anomaly_score_map: (N, H, W) anomaly score maps
        max_step: number of thresholds
        expect_fpr: maximum false positive rate

    Returns:
        PRO AUC score in percentage
    """

    # Thresholds
    min_th = anomaly_score_map.min()
    max_th = anomaly_score_map.max()
    thresholds = np.linspace(min_th, max_th, max_step)

    # Precompute regions once (MAJOR speed-up)
    regions_per_image = [
        [region.coords for region in regionprops(label(mask))]
        for mask in gt_mask
    ]

    inverse_masks = 1 - gt_mask
    inv_sum = inverse_masks.sum()

    pros_mean = []
    fprs = []

    for th in thresholds:
        binary_score_maps = anomaly_score_map > th

        pro = []
        for binary_map, regions in zip(binary_score_maps, regions_per_image):
            for coords in regions:
                tp_pixels = binary_map[coords[:, 0], coords[:, 1]].sum()
                pro.append(tp_pixels / len(coords))

        pros_mean.append(np.mean(pro) if len(pro) > 0 else 0.0)

        fprs.append(
            np.logical_and(inverse_masks, binary_score_maps).sum() / inv_sum
        )

    pros_mean = np.asarray(pros_mean)
    fprs = np.asarray(fprs)

    # FPR filtering and rescaling
    idx = fprs < expect_fpr
    fprs = (fprs[idx] - fprs[idx].min()) / (fprs[idx].max() - fprs[idx].min())
    pros_mean = pros_mean[idx]

    return auc(fprs, pros_mean) * 100

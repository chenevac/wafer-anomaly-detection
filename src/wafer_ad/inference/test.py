from typing import Dict, Tuple
import numpy as np
import torch
from tqdm import tqdm

from wafer_ad.training.metric import compute_imagewise_retrieval_metrics, compute_pixelwise_retrieval_metrics


def test(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    enable_progress_bar: bool = True,
    device: str = "cpu",
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    assert not dataloader.dataset.is_for_train, "Dataloader should be for test/evaluation, not for training."
    model.eval()
    model.to(device)
    if enable_progress_bar:
        dataloader = tqdm(dataloader, total=len(dataloader))
    gt_label_list = list()
    gt_mask_list = list()
    anomaly_score_list = list()
    anomaly_score_map_add_list = list() 
    anomaly_score_map_mul_list = list()
    with torch.no_grad():
        for (images, labels, masks) in dataloader:
            images = images.to(device)
            z_list, jac = model(images)
            anomaly_score, anomaly_score_map_add, anomaly_score_map_mul = model.post_process_forward(z_list, jac)
            
            anomaly_score_list.extend(anomaly_score)
            anomaly_score_map_add_list.extend(anomaly_score_map_add)
            anomaly_score_map_mul_list.extend(anomaly_score_map_mul)
            gt_label_list.extend(labels.cpu().data.numpy())
            gt_mask_list.extend(masks.cpu().data.numpy())
    gt_label = np.asarray(gt_label_list, dtype=np.bool)
    gt_mask = np.squeeze(np.asarray(gt_mask_list, dtype=np.bool), axis=1)
    anomaly_score_map_add_list = np.asarray(anomaly_score_map_add_list)
    anomaly_score_map_mul_list = np.asarray(anomaly_score_map_mul_list)
    
    return {
        "imagewise_retrieval_metrics": compute_imagewise_retrieval_metrics(anomaly_score_list, gt_label),
        "pixelwise_retrieval_metrics_add": compute_pixelwise_retrieval_metrics(anomaly_score_map_add_list, gt_mask),
        "pixelwise_retrieval_metrics_mul": compute_pixelwise_retrieval_metrics(anomaly_score_map_mul_list, gt_mask),
    }
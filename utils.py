import torch
import torch.nn.functional as F


def psnr_mse(input, target):
    mse = F.mse_loss(input, target)
    psnr = 20.0 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr


def psnr_mae(input, target):
    mae = F.l1_loss(input, target)
    psnr = 20.0 * torch.log10(1.0 / torch.sqrt(mae))
    return psnr


def compute_ts_road_map(road_map1, road_map2):
    """Computes the mean threat score of road images for an entire batch"""
    tp = (road_map1 * road_map2).sum(axis=(1, 2))
    ts = tp * 1.0 / (road_map1.sum(axis=(1, 2)) + road_map2.sum(axis=(1, 2)) - tp)
    return ts.mean()


LOSS = {
    "bce": F.binary_cross_entropy_with_logits,
    "mse": F.mse_loss,
    "mae": F.l1_loss,
    "psnr_mse": psnr_mse,
    "psnr_mae": psnr_mae,
}

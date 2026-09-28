
import torch
import numpy as np


class PSNR:
    """Peak Signal to Noise Ratio (ref 코드의 PSNR에서 FID 의존성만 뺀 버전)
    이미지를 [0, 255]로 바꾼 뒤 이미지별 PSNR을 계산한다."""

    def __init__(self, reduction='mean'):
        self.name = "PSNR"
        self.reduction = reduction

    def __call__(self, img1, img2, min_val=0., max_val=1., **kwargs):
        with torch.no_grad():
            img1 = torch.clamp(img1, min_val, max_val)
            img1 = (img1 - min_val) / (max_val - min_val) * 255.
            img2 = torch.clamp(img2, min_val, max_val)
            img2 = (img2 - min_val) / (max_val - min_val) * 255.
            img1, img2 = img1.round(), img2.round()
            mse = torch.mean((img1 - img2) ** 2, dim=(1, 2, 3)).clamp(min=1e-10)
            psnr = (10 * torch.log10(255.0 ** 2 / mse)).cpu().numpy()
        if self.reduction == 'mean':
            return float(np.mean(psnr))
        elif self.reduction == 'sum':
            return float(np.sum(psnr))
        elif self.reduction == 'none':
            return psnr
        raise NotImplementedError()

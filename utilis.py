from pytorch_msssim import MS_SSIM
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps).mean()
        return loss

class SSIMLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0.5):
        super().__init__()
        self.ms_ssim = MS_SSIM(data_range=1.0, win_size=5, size_average=True, channel=1)

    def forward(self, pred, target):
        ms_ssim_loss = 1 - self.ms_ssim(pred, target)
        return ms_ssim_loss

#VGG Loss vascular-7
class VGGFeatureExtractor(nn.Module):
    def __init__(self,layer_index=7):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features[:layer_index]
        for param in vgg.parameters():
            param.requires_grad_(False)
        self.vgg = vgg.eval()

    def forward(self, x):
        x = torch.cat([x] * 3, dim=1)  # [B,3,H,W]

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x_norm = (x - mean) / std

        features = self.vgg(x_norm)
        return features

class PSNRCalculator(nn.Module):
    def __init__(self, data_range=1.0):
        super(PSNRCalculator, self).__init__()
        self.data_range = data_range

    def forward(self, pred, target):

        mse = F.mse_loss(pred, target, reduction='none').mean(dim=[1, 2, 3])
        mse = torch.clamp(mse, min=1e-10)
        psnr = 10.0 * torch.log10((self.data_range ** 2) / mse)  # [B]

        return psnr.mean()


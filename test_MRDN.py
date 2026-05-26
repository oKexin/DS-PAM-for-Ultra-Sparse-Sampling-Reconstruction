import matplotlib.pyplot as plt
from model import MRDN
import torch
import cv2
import numpy as np
from torchvision import transforms
from utilis import PSNRCalculator
from torchmetrics.image import StructuralSimilarityIndexMeasure
import lpips
import warnings
import os
warnings.filterwarnings("ignore", category=UserWarning)

def normalized(image):
    return (image - image.min()) / (image.max() - image.min())

def dowmsampling(image, ratio, flag = True):
    h, w = image.shape[:2]
    if flag:
        lr_patch = image[::ratio, ::ratio]
    else:
        lr_patch = np.zeros((h, w), dtype=image.dtype)
        lr_patch[::ratio, ::ratio] = image[::ratio, ::ratio]
    return lr_patch



if __name__ == '__main__':
    scale_factor = 16
    saveflag = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = MRDN(scale_factor=scale_factor).to(device)
    generator.eval()
    generator.load_state_dict(torch.load('Result/proposed_{}x.pth'.format(scale_factor), map_location=device))
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    lr_frames = []
    num = 4
    or_path = 'testdata/{}_OR.png'.format(num)
    ar_path = 'testdata/AR_Enh_{}x/{}_Enh.png'.format(scale_factor, num)
    or_frame = cv2.imread(or_path, 0)
    ar_frame = cv2.imread(ar_path, 0)

    or_frame = normalized(or_frame).astype(np.float32)
    ar_frame = normalized(ar_frame).astype(np.float32)
    lor_frame = dowmsampling(or_frame, scale_factor, True)
    lor_frame = normalized(lor_frame).astype(np.float32)
    plt.figure()
    plt.imshow(lor_frame, cmap='hot')
    plt.axis('off')
    if saveflag:
        plt.savefig('Result/{}x/LR.png'.format(scale_factor), dpi=300, bbox_inches='tight', pad_inches=0)
    lor_frame = transform(lor_frame)
    ar_frame = transform(ar_frame)
    lor_frame = lor_frame.unsqueeze(0)
    ar_frame = ar_frame.unsqueeze(0)
    lor_frame = lor_frame.to(device)
    ar_frame = ar_frame.to(device)
    with torch.no_grad():
        hr_pred = generator(lor_frame,ar_frame)
    result = hr_pred.squeeze().detach().cpu().numpy()
    result = normalized(result).astype(np.float32)
    index_result = transform(result).to(device)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(or_frame, cmap='hot')
    axes[1].imshow(result, cmap='hot')
    plt.show()
    plt.figure()
    plt.imshow(or_frame, cmap='hot')
    plt.axis('off')
    if saveflag:
        plt.savefig('HR-{}.png'.format(num), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.figure()
    plt.imshow(result, cmap='hot')
    plt.axis('off')
    if saveflag:
        plt.savefig('Result/{}x/proposed.png'.format(scale_factor), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()
    #Count Index
    or_frame = transform(or_frame).to(device)
    lpips_loss_fn = lpips.LPIPS(net='vgg', verbose=False).to(device)
    lpips_value = lpips_loss_fn(index_result.unsqueeze(0), or_frame.unsqueeze(0)).item()
    psnr_count = PSNRCalculator().to(device)
    ssim_count = StructuralSimilarityIndexMeasure(data_range=1.0, reduction=None).to(device)
    psnr = psnr_count(index_result.unsqueeze(0), or_frame.unsqueeze(0)).item()
    ssim = ssim_count(index_result.unsqueeze(0), or_frame.unsqueeze(0)).item()
    print(f"SSIM: {ssim:.4f}, PSNR: {psnr:.2f}, LPIPS: {lpips_value:.4f}")

    if saveflag:
        image_uint8 = (result * 255).astype(np.uint8)
        save_path = 'Result/{}x/raw_proposed.png'.format(scale_factor)
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except PermissionError:
                pass
        cv2.imwrite(save_path, image_uint8)



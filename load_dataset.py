import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import random
import re
import os

#AR-OR-use

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

def is_high_frequency_patch(img_patch, brightness_threshold=3, variance_threshold1=10, variance_threshold2=200):

    brightness_mean = np.mean(img_patch)
    brightness_var = np.var(img_patch)


    if (brightness_mean < brightness_threshold or brightness_var < variance_threshold1 or
            (brightness_mean< 10 and brightness_var > variance_threshold2)):
        return False, {
            'brightness_mean': brightness_mean,
            'brightness_var': brightness_var,
        }
    return True, {
        'brightness_mean': brightness_mean,
        'brightness_var': brightness_var,
    }

def random_rotate(patch, angle):
    if angle == 90:
        return cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(patch, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return patch

def random_flip(patch, flip_type):
    if flip_type is None:
        return patch
    return cv2.flip(patch, flip_type)

def add_gaussian_noise(patch, sigma):
    gauss = np.random.normal(0, sigma, patch.shape)
    noisy_patch = patch + gauss
    noisy_patch = np.clip(noisy_patch, 0, 1)
    return noisy_patch

class ORARDataset(Dataset):
    def __init__(self, __data_dir__, __enh_dir__, __scale_factor__, __patch_size__, d_flag):
        self.dir = __data_dir__
        self.enh_dir = __enh_dir__
        self.scale_factor = __scale_factor__
        self.patch_size = __patch_size__
        self.lr_patch_size = __patch_size__ // __scale_factor__
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.downsample_flag = d_flag
        self.file_count = self._count_valid_files()

    def _count_valid_files(self):

        if not os.path.exists(self.dir):
            raise FileNotFoundError(f"no file: {self.dir}")

        pattern = re.compile(r'^\d+_OR\.png$')
        count = 0
        for filename in os.listdir(self.dir):
            if pattern.match(filename):
                count += 1
        return count

    def __getitem__(self, idx):

        or_frame = cv2.imread(f"{self.dir}/{idx + 1}_OR.png", 0)
        ar_enhance = cv2.imread(f"{self.enh_dir}/{idx + 1}_Enh.png", 0)
        h, w = or_frame.shape

        while True:
            h_start = np.random.randint(0, h - self.patch_size + 1)
            w_start = np.random.randint(0, w - self.patch_size + 1)
            or_patch = or_frame[h_start:h_start + self.patch_size, w_start:w_start + self.patch_size]
            is_high_freq, _ = is_high_frequency_patch(or_patch)
            if is_high_freq:
                break
        lr_h_start = h_start // self.scale_factor
        hr_h_start = lr_h_start * self.scale_factor
        lr_w_start = w_start // self.scale_factor
        hr_w_start = lr_w_start * self.scale_factor
        or_patch = or_frame[hr_h_start:hr_h_start + self.patch_size, hr_w_start:hr_w_start + self.patch_size]
        angle = random.choice([0, 90, 180, 270])
        flip_type = random.choice([-1, 0, 1, None])
        sigma = np.random.uniform(0.001, 0.025)

        or_patch = random_rotate(or_patch, angle)
        or_patch = random_flip(or_patch, flip_type)
        or_patch = normalized(or_patch).astype(np.float32)
        lor_patch = dowmsampling(or_patch, self.scale_factor, self.downsample_flag)
        lor_patch = add_gaussian_noise(lor_patch,sigma)
        lor_patch = normalized(lor_patch).astype(np.float32)

        enh_patch = ar_enhance[lr_h_start:lr_h_start + self.lr_patch_size, lr_w_start:lr_w_start + self.lr_patch_size]
        enh_patch = random_rotate(enh_patch, angle)
        enh_patch = random_flip(enh_patch, flip_type)
        enh_patch = normalized(enh_patch).astype(np.float32)
        lor_patch = self.transform(lor_patch)
        enh_patch = self.transform(enh_patch)
        or_patch = self.transform(or_patch)
        return lor_patch, enh_patch, or_patch

    def __len__(self):
        return self.file_count

if __name__ == "__main__":
    data_dir  = 'dataset/'
    enh_dir = 'AR_Enh_12x/'
    train_dataset = ORARDataset(__data_dir__=data_dir, __enh_dir__=enh_dir, __scale_factor__=12, __patch_size__=192, d_flag=True)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=1)
    print(len(train_loader))
    batch_idx, (lor_data, lar_data, or_data) = list(enumerate(train_loader))[0]
    fig, axes = plt.subplots(2, 3)
    for epoch in range(2):
        for batch_idx,(lor_data, enh_data, or_data) in enumerate(train_loader):
            if batch_idx == 1:
                # print(lor_data.shape)
                # print(or_data.shape)
                axes[epoch][0].imshow(lor_data.squeeze()[0])
                axes[epoch][1].imshow(enh_data.squeeze()[0])
                axes[epoch][2].imshow(or_data.squeeze()[0])
            else:
                pass
    plt.show()

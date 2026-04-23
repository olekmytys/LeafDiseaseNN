import os
from torchvision.io import decode_image
from torch.utils.data import Dataset
import pywt
import numpy as np

import torch

class ImageDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.root = dir
        self.transform=transform
        self.class_names = sorted(os.listdir(dir))
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        self.img_labels = []
        for class_name in self.class_names:
            class_dir = os.path.join(self.root, class_name)
            for f in os.listdir(class_dir):
                if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".webp")):
                    path = os.path.join(class_dir, f)
                    label = self.class_to_idx[class_name]
                    self.img_labels.append((path, label))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        image = decode_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label

class WaveletTransform:
    def __init__(self, type='sym4', level=2):
        self.wavelet = type
        self.level = level

    def __call__(self, image):
        result = torch.zeros_like(image)

        for c in range(3):
            image_np = image[c].numpy()

            coeffs = pywt.wavedec2(image_np, wavelet=self.wavelet, level=self.level)
            approx = coeffs[0]

            approx = torch.tensor(approx, dtype=torch.float32)
            approx = approx.unsqueeze(0).unsqueeze(0) # dodaje bathc i kanal [1, 1, H/2, W/2]
            approx = torch.nn.functional.interpolate(approx, size=image.shape[1:], mode='bilinear', align_corners=False)
            
            result[c] = approx.squeeze()
        return result    

# !!! TRZEBA SPRAWDZIC POKI CO DZIALA TAK SOBIE !!!
class FourierSharpening:
    def __init__(self, radius=30, strength=0.7):
        self.radius = radius
        self.strength = strength

    def __call__(self, image):
        _, h, w = image.shape
        cy, cx = h // 2, w // 2
        mask = torch.ones(h, w)
        mask[cy-self.radius:cy+self.radius, cx-self.radius:cx+self.radius] = 0

        edges = torch.zeros_like(image)
        for c in range(3):
            fft = torch.fft.fft2(image[c])
            fft_shift = torch.fft.fftshift(fft)
            fft_shift *= mask
            ifft = torch.fft.ifftshift(fft_shift)
            edges[c] = torch.abs(torch.fft.ifft2(ifft))

        return (image + self.strength * edges).clamp(0, 1)




import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 128

# 처리용
def load_image(path, size=128):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    x = np.asarray(img).astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0)
    return x

def show_rgb(x, title=""):
    if x.dim() == 4:
        x = x[0]
    x = x.detach().cpu().clamp(0,1).permute(1,2,0).numpy()
    plt.figure(figsize=(4,4))
    plt.imshow(x)
    plt.axis("off")
    plt.title(title)
    plt.show()

def show_feature(feat1, title1, feat2, title2, max_ch=16):
    rows = 4
    cols = 9

    plt.figure(figsize=(10, 5))

    n1 = min(feat1.shape[1], max_ch)
    for i in range(n1):
        r = i // 4
        c = i % 4
        idx = r * cols + c + 1

        plt.subplot(rows, cols, idx)
        plt.imshow(feat1[0, i].detach().cpu().numpy(), cmap="gray")
        plt.axis("off")

    n2 = min(feat2.shape[1], max_ch)
    for i in range(n2):
        r = i // 4
        c = i % 4
        idx = r * cols + (c + 5) + 1

        plt.subplot(rows, cols, idx)
        plt.imshow(feat2[0, i].detach().cpu().numpy(), cmap="gray")
        plt.axis("off")

    plt.subplots_adjust(left=0.05, right=0.95)

    plt.figtext(0.25, 0.92, title1, ha='center', fontsize=14)
    plt.figtext(0.75, 0.92, title2, ha='center', fontsize=14)

    plt.show()

# 인코더 모델 정의
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU()
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        outs = {}
        x = self.enc1(x); outs["enc1"] = x
        x = self.enc2(x); outs["enc2"] = x
        x = self.enc3(x); outs["enc3"] = x
        x = self.enc4(x); outs["latent"] = x
        return outs

# 시각화 부분
def visualize(image_path, ch_to_show=0, grid_ch=16):
    model = Encoder().to(DEVICE)
    model.eval()

    x = load_image(image_path, IMG_SIZE).to(DEVICE)
    outs = model(x)

    show_rgb(x, "Input")

    show_feature(outs["enc1"], "enc1", outs["enc2"], "enc2")
    show_feature(outs["enc3"], "enc3", outs["latent"], "latent")


IMG_PATH = "../resource/SpeakIncoder.jpg"
visualize(IMG_PATH, ch_to_show=0, grid_ch=16)

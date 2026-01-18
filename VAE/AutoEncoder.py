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

def show_feature(feat, max_ch=16, title=""):
    C = feat.shape[1]
    n = min(C, max_ch)
    cols = int(math.sqrt(n))
    rows = int(math.ceil(n / cols))

    plt.figure(figsize=(2.2*cols, 2.2*rows))
    for i in range(n):
        plt.subplot(rows, cols, i+1)
        plt.imshow(feat[0, i].detach().cpu().numpy(), cmap="gray")
        plt.axis("off")
        plt.title(f"ch {i}", fontsize=9)

    plt.suptitle(f"{title}\nshape={tuple(feat.shape)}", y=1.02)
    plt.tight_layout()
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

    for name, feat in outs.items():
        show_feature(feat, max_ch=grid_ch, title=name)


IMG_PATH = "../resource/SpeakIncoder.jpg"
visualize(IMG_PATH, ch_to_show=0, grid_ch=16)

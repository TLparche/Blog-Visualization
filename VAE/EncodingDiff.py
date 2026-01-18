import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 128


# 처리용
def load_image(path, size=128):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    x = np.asarray(img).astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
    return x


def show_rgb(x, title=""):
    if x.dim() == 4:
        x = x[0]
    x = x.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    plt.figure(figsize=(4, 4))
    plt.imshow(x)
    plt.axis("off")
    plt.title(title)
    plt.show()

def show_comparison(original, reconstructed, title="Result"):
    org = original[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    rec = reconstructed[0].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()

    plt.figure(figsize=(10, 5))

    # 원본 이미지
    plt.subplot(1, 2, 1)
    plt.imshow(org)
    plt.title("Original Image")
    plt.axis("off")

    # 복원된 이미지
    plt.subplot(1, 2, 2)
    plt.imshow(rec)
    plt.title("Reconstructed Image")
    plt.axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
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
class AutoEncoder(nn.Module):
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

        self.dec4 = (nn.Sequential
                     (nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
                      nn.ReLU())
                     )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        z = self.enc4(e3)

        d4 = self.dec4(z)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        out = self.dec1(d2)

        return out, {"enc1": e1, "enc2": e2, "enc3": e3, "latent": z}


# 시각화 부분
def visualize(image_path):
    model = AutoEncoder().to(DEVICE)

    x = load_image(image_path, IMG_SIZE).to(DEVICE)

    # 학습 전 모델
    with torch.no_grad():
        _, outs_untrained = model(x)

    show_rgb(x, "Original Input")
    show_feature(outs_untrained["enc1"], "Untrained Enc1", outs_untrained["enc2"], "Untrained Enc2")
    show_feature(outs_untrained["enc3"], "Untrained Enc3", outs_untrained["latent"], "Untrained Latent")

    # 학습 (일부러 과적합)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(300):
        optimizer.zero_grad()
        recon, _ = model(x)
        loss = criterion(recon, x)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.5f}")

    # 학습 후 모델
    model.eval()
    with torch.no_grad():
        reconstruct_img, outs_trained = model(x)

    show_feature(outs_trained["enc1"], "Trained Enc1", outs_trained["enc2"], "Trained Enc2")
    show_feature(outs_trained["enc3"], "Trained Enc3", outs_trained["latent"], "Trained Latent")

    x = load_image(image_path, IMG_SIZE).to(DEVICE)
    show_comparison(x, reconstruct_img)


IMG_PATH = "../resource/SpeakIncoder.jpg"
visualize(IMG_PATH)

import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Linear Beta Schedule 설정
def get_scheduler(timesteps=1000, start=0.0001, end=0.02):
    betas = torch.linspace(start, end, timesteps)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod

# 이미지 받아서 노이즈 생성
def forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    noise = torch.randn_like(x_0)  # 정규분포 노이즈 생성

    # 시점 맞게
    sqrt_alpha_t = extract(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alpha_t = extract(sqrt_one_minus_alphas_cumprod, t, x_0.shape)

    # 공식 적용
    return sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise

# reshape용
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# 이미지 나열
def show_images(images, title="Fixed Forward Diffusion Process"):
    plt.figure(figsize=(10, 2))
    plt.suptitle(title, fontsize=16)

    for i, img in enumerate(images):
        # 텐서 이미지 변환
        img = img.squeeze().permute(1, 2, 0).cpu().numpy()
        img = (img + 1) / 2
        img = np.clip(img, 0, 1)  

        plt.subplot(1, len(images), i + 1)
        plt.imshow(img)
        plt.axis('off')

    plt.show()


# 이미지 불러오기
image_path = "../resource/DiffusionProcessExample.png"  # 파일 경로

image = Image.open(image_path).convert("RGB")

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 크기 조정
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1)
])

x_0 = transform(image).unsqueeze(0)

# 스케줄러 설정
T = 1000  # 총 타임스텝
sqrt_alphas, sqrt_one_minus_alphas = get_scheduler(timesteps=T)

# 타임스텝 설정
plot_steps = [0, 50, 100, 200, 400, 700, 999]
results = []

for step in plot_steps:
    t = torch.tensor([step])
    # 노이즈 추가
    x_t = forward_diffusion_sample(x_0, t, sqrt_alphas, sqrt_one_minus_alphas)
    results.append(x_t)

# 시각화
show_images(results)
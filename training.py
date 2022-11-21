# %%
import src
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from datasets import load_dataset
from PIL import Image
import requests
from pathlib import Path
from torch.optim import Adam


# %%
beta_start = 0.0001
beta_end = 0.02
timesteps = 1000

scheduler = src.LinearScheduler(beta_start=beta_start, beta_end=beta_end, timesteps=timesteps)

# %%
reverse_transform = Compose([
     Lambda(lambda t: (t + 1) / 2),
     Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
     Lambda(lambda t: t * 255.),
     Lambda(lambda t: t.numpy().astype(np.uint8)),
     ToPILImage(),
])

forward_diffusion = src.ForwardDiffusion(sqrt_alphas_cumprod=scheduler.sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=scheduler.sqrt_one_minus_alphas_cumprod, reverse_transform=reverse_transform)

image_size = 28
num_channels = 1
batch_size = 128
dataset_name = 'fashion_mnist'
num_workers = 0
dataset = load_dataset(dataset_name, num_proc=num_workers)

# %%
forward_transform = Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
])

def transforms(examples):
   examples["pixel_values"] = [forward_transform(image.convert("L")) for image in examples["image"]]
   del examples["image"]

   return examples

transformed_dataset = dataset.with_transform(transforms).remove_columns("label")

dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers)

# %%
results_folder_name = './results'
sample_and_save_freq = 100
results_folder = Path(results_folder_name)

# %%
device = "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu"
device = torch.device(device)

# %%
model = src.DDPM(n_features=image_size, in_channels=num_channels, channel_scale_factors=(1, 2, 4,))
model.to(device)

# %%
learninig_rate = 1e-3
optimizer = Adam(model.parameters(), lr=learninig_rate)
criterion = src.get_loss

# %%
timesteps = 200
sampler = src.Sampler(betas=scheduler.betas, sqrt_one_minus_alphas_cumprod=scheduler.sqrt_one_minus_alphas_cumprod, sqrt_one_by_alphas=scheduler.sqrt_one_by_alphas, posterior_variance=scheduler.posterior_variance, timesteps=timesteps)

# %%
epochs = 5

src.train(
    image_size=image_size, 
    num_channels=num_channels, 
    epochs=epochs, 
    timesteps=timesteps, 
    sample_and_save_freq=sample_and_save_freq, 
    save_folder=results_folder, 
    forward_diffusion_model=forward_diffusion, 
    denoising_model=model, 
    criterion=criterion, 
    optimizer=optimizer, 
    dataloader=dataloader, 
    sampler=sampler, 
    device=device
)

# %%




import torch
from torchvision.utils import save_image
from ..utils import visualize_loss, save_image_grid, save_model
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def train(image_size, num_channels, epochs, timesteps, sample_and_save_freq, save_folder, forward_diffusion_model, denoising_model, criterion, optimizer, dataloader, sampler, device):
    save_folder.mkdir(exist_ok = True)
    writer = SummaryWriter('runs')

    loss_type="huber"
    for epoch in range(epochs):
        with tqdm(dataloader, desc=f'Training DDPM') as pbar:
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()

                batch_size = batch["pixel_values"].shape[0]
                batch = batch["pixel_values"].to(device)

                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                t = torch.randint(0, timesteps, (batch_size,), device=device).long()
                loss = criterion(forward_diffusion_model=forward_diffusion_model, denoising_model=denoising_model, x_start=batch, t=t, loss_type=loss_type)
                # if step % 100 == 0:
                #     print(f"Epoch {epoch} Loss: {loss.item()}")
                loss.backward()
                optimizer.step()

                pbar.set_postfix(Loss=f"{loss:.4f}")
                pbar.update()

                visualize_loss(writer=writer, loss=loss, step=(epoch*len(dataloader) + step), loss_type=loss_type)

                # save generated images
                if step != 0 and step % sample_and_save_freq == 0:
                    samples = sampler.sample(model=denoising_model, image_size=image_size, batch_size=batch_size, channels=num_channels)
                    all_images = samples[-1] 
                    all_images = (all_images + 1) * 0.5
                    save_image_grid(step=(step*len(dataloader) + step), writer=writer, n_images=5, samples=all_images)
        save_model(step=epoch, model=denoising_model)


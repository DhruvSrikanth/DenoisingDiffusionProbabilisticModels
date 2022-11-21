import torch
from torchvision.utils import save_image
from ..utils import num_to_groups


def train(image_size, num_channels, epochs, timesteps, sample_and_save_freq, save_folder, forward_diffusion_model, denoising_model, criterion, optimizer, dataloader, sampler, device):
    save_folder.mkdir(exist_ok = True)

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()

            batch_size = batch["pixel_values"].shape[0]
            batch = batch["pixel_values"].to(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = criterion(forward_diffusion_model=forward_diffusion_model, denoising_model=denoising_model, x_start=batch, t=t, loss_type="huber")

            if step % 100 == 0:
                print(f"Epoch {epoch} Loss: {loss.item()}")

            loss.backward()
            optimizer.step()

            # save generated images
            if step != 0 and step % sample_and_save_freq == 0:
                milestone = step // sample_and_save_freq
                batches = num_to_groups(4, batch_size)
                all_images_list = list(map(lambda n: sampler.sample(model=denoising_model, image_size=image_size, batch_size=n, channels=num_channels), batches))
                all_images = torch.cat(all_images_list, dim=0)
                all_images = (all_images + 1) * 0.5
                save_image(all_images, str(save_folder / f'sample-{milestone}.png'), nrow = 6)
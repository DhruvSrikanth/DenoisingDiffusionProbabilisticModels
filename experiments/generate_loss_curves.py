import numpy as np
import pandas as pd
import src
import torch
from torchvision.transforms import Compose, Lambda, ToPILImage
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

def transforms(examples):
        forward_transform = Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda t: (t * 2) - 1)
        ])
        
        examples["pixel_values"] = [forward_transform(image.convert("L")) for image in examples["image"]]
        del examples["image"]

        return examples

def get_dataloaders(dataset_name, batch_size):
    dataset = load_dataset(dataset_name)

    transformed_dataset = dataset.with_transform(transforms).remove_columns("label")

    dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(transformed_dataset["test"], batch_size=batch_size, shuffle=True)

    return dataloader, val_dataloader


def get_model_loss(dataloader, timesteps, forward_diffusion_model, denoising_model, criterion, loss_type, device):
    denoising_model.eval()
    total_loss = 0
    total_size = 0
    with tqdm(dataloader, desc=f'Getting DDPM Loss') as pbar:
        for step, batch in enumerate(dataloader):
            batch_size = batch["pixel_values"].shape[0]
            batch = batch["pixel_values"].to(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            loss = criterion(forward_diffusion_model=forward_diffusion_model, denoising_model=denoising_model, x_start=batch, t=t, loss_type=loss_type)
            total_loss += loss.item()
            total_size += 1
            pbar.set_postfix(Epoch=f"{step+1}/{len(dataloader)}", Loss=f"{loss:.4f}")
            pbar.update()
    
    return total_loss / total_size


def get_losses(models_path, epochs, dataloader, timesteps, forward_diffusion_model, model, criterion, loss_type, device):
    losses = []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}:")
        state = torch.load(f"{models_path}/model_{epoch + 1}.pth", map_location=device)
        model.load_state_dict(state)
        model.to(device)

        loss = get_model_loss(dataloader=dataloader, timesteps=timesteps, forward_diffusion=forward_diffusion_model, denoising_model=model, criterion=criterion, loss_type=loss_type, device=device)
        losses.append(loss)
    return losses

def determine_scheduler(beta_start, beta_end, timesteps, scheduler_choice):
    if scheduler_choice == "linear":
        scheduler = src.LinearScheduler(beta_start=beta_start, beta_end=beta_end, timesteps=timesteps)
    elif scheduler_choice == "quadratic":
        scheduler = src.QuadraticScheduler(beta_start=beta_start, beta_end=beta_end, timesteps=timesteps)
    elif scheduler_choice == "sigmoid":
        scheduler = src.SigmoidScheduler(beta_start=beta_start, beta_end=beta_end, timesteps=timesteps)
    elif scheduler_choice == "cosine":
        scheduler = src.CosineScheduler(beta_start=beta_start, beta_end=beta_end, timesteps=timesteps)
    return scheduler



def get_loss_experiment(root_path, scheduler_choice):
    batch_size = 128
    dataset_name = "cifar10"
    train_dataloader, val_dataloader = get_dataloaders(dataset_name=dataset_name, batch_size=batch_size)

    device = "cpu"
    device = torch.device(device)

    # root_path = "../runs"
    models_path = f"{root_path}/{scheduler_choice}"
    beta_start = 0.0001
    beta_end = 0.02
    timesteps = 300
    scheduler = determine_scheduler(beta_start=beta_start, beta_end=beta_end, timesteps=timesteps, scheduler_choice=scheduler_choice)


    reverse_transform = Compose([
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])
    forward_diffusion = src.ForwardDiffusion(sqrt_alphas_cumprod=scheduler.sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=scheduler.sqrt_one_minus_alphas_cumprod, reverse_transform=reverse_transform)
    
    image_size = 32
    num_channels = 3
    model = src.DDPM(n_features=image_size, in_channels=num_channels, channel_scale_factors=(1, 2, 4,))

    criterion = src.get_loss


    epochs = 50
    loss_type="huber"
    train_losses = get_losses(
        models_path=models_path, 
        epochs=epochs, dataloader=train_dataloader, 
        timesteps=timesteps, 
        forward_diffusion_model=forward_diffusion, 
        model=model, 
        criterion=criterion, 
        loss_type=loss_type, 
        device=device
    )

    val_losses = get_losses(
        models_path=models_path,
        epochs=epochs, dataloader=val_dataloader,
        timesteps=timesteps,
        forward_diffusion_model=forward_diffusion,
        model=model,
        criterion=criterion,
        loss_type=loss_type,
        device=device
    )

    df = pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses})
    df.to_csv(f"{root_path}/{scheduler_choice}/losses.csv", index=False)
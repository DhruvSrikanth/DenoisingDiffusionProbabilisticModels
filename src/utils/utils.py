import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import save_image
from torchvision.utils import make_grid
import matplotlib.animation as animation

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract_time_index(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

# source: https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
def plot(image, imgs, with_orig=False, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(figsize=(200,200), nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [image] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()


def visualize_loss(step, loss, loss_type, writer):
    writer.add_scalar(f'{loss_type} loss', loss, step)


def save_image_grid(step, writer, n_images, samples):
    n_square_images = torch.tensor(samples[:n_images**2])
    save_image(n_square_images, f"results/sample_{step}.png", nrow=n_images, normalize=True)

    # Read in and add to tensorboard
    img_grid = read_image(f"results/sample_{step}.png", mode=torchvision.io.ImageReadMode.GRAY)

    writer.add_image(f'Generated Images', img_grid, global_step=step)

def create_gif(samples, n_images, image_size, num_channels, timesteps):
    fig = plt.figure()
    img_grids = []
    for i in range(timesteps):
        n_square_images = torch.tensor(samples[i][:n_images**2].reshape(n_images**2, num_channels, image_size, image_size))
        image_grid = make_grid(n_square_images, nrow=n_images, padding=2, pad_value=1, normalize=True)
        img_grid = plt.imshow(image_grid.permute(1, 2, 0), animated=True)
        img_grids.append([img_grid])

    plt.title("DDPM Generated Samples")
    plt.axis("off")
    plt.tight_layout()

    animate = animation.ArtistAnimation(fig, img_grids, interval=5, blit=True, repeat_delay=1000)
    animate.save('diffusion.gif')
    plt.show()


def save_model(epoch, model):
    torch.save(
        model.state_dict(),
        f"runs/model_{epoch + 1}.pth"
    )

def load_model(epoch, model, device):
    state = torch.load(f"runs/model_{epoch + 1}.pth", map_location=device)
    model.load_state_dict(state)
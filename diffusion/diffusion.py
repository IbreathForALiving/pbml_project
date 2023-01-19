# pylint: disable=no-member
""" Module for precomputed variables needed in a diffusion model"""
from typing import Tuple, List, Optional
import torch
import torch.nn.functional as F
from torchtyping import TensorType
from torchvision import datasets
from .model import Unet
import matplotlib.pyplot as plt
from . import DEVICES

class PreComputedVariables:
    """
    Precomputes all the variables needed for the diffusion model.
    Or atleast all the variables which are on closed form
    """
    def __init__(self, times: int = 300, beta_start: float = 0.0001, beta_end: float = 0.02) -> None:
        self.times = times

        # Compute betas
        self.betas: torch.Tensor = self.precompute_beta_schedule(self.times, beta_start, beta_end)

        # Compute all the different alpha terms
        self.alphas = 1. - self.betas
        alphas_bar = torch.cumprod(self.alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar[:-1], (1, 0), value=1.0)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_bar = torch.sqrt(alphas_bar)

        self.sqrt_one_minus_alphas_bar = torch.sqrt(1. - alphas_bar)
        self.posterior_variance = self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar)

    @staticmethod
    def precompute_beta_schedule(times: int, start: float, end: float) -> TensorType[float]:
        """
        Precomputes the beta values based on the
        number of time steps and the start and end values of the generated beta values.
        """
        return torch.linspace(start, end, times)

    @staticmethod
    def get_index_from_list(vals, time, x_shape): # TODO
        """
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = time.shape[0]
        out = vals.gather(-1, time.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(time.device)

    @staticmethod
    def epsilon(x_0: TensorType[float]) -> TensorType[float]:
        """ Noise added to the image """
        return torch.randn_like(x_0)

    def forward_sample(
        self,
        x_0: TensorType["batch", "channel", "x_size", "y_size", float],
        time: TensorType["batch", torch.long],
        device: DEVICES = "cpu"
        ) -> Tuple[TensorType["batch", "channel", "x_size", "y_size", float], TensorType["batch", "channel", "x_size", "y_size", float]]:
        """
        Applies the forward process to an image (Adds noise to the image).
        """
        # epsilon
        noise: torch.Tensor = self.epsilon(x_0).to(device)
        # sqrt(alpha)
        sqrt_alphas_bar_t = self.get_index_from_list(self.sqrt_alphas_bar, time, x_0.shape)

        #sqrt(1 - alpha)
        sqrt_one_minus_alphas_bar_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_bar, time, x_0.shape
        )
        # Compute x_t
        x_t: torch.Tensor = (
            sqrt_alphas_bar_t.to(device) * x_0.to(device)
            + sqrt_one_minus_alphas_bar_t.to(device) * noise
        )
        return x_t, noise
    
    def time_sample(self, batch_size: int, device: DEVICES = "cpu") -> TensorType[torch.long]:
        """ Sample different times"""
        return torch.randint(0, self.times, (batch_size,), device=device).long()
    
    @torch.no_grad()
    def generate_sample(self, model: Unet, x: TensorType["batch", "channel", "x_size", "y_size", float], timestep: TensorType["batch", torch.long]):
        
        # Get precomputed values
        betas_t = self.get_index_from_list(self.betas, timestep, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_bar, timestep, x.shape
        )
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, timestep, x.shape)
        
        # Get epsilon_pred (epsilon_theta)
        epsilon_pred = model(x, timestep)

        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (
            x - epsilon_pred * betas_t / sqrt_one_minus_alphas_cumprod_t
        )
        
        if timestep == 0:
            return model_mean
        
        posterior_variance_t = self.get_index_from_list(self.posterior_variance, timestep, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

def show_images(dataset: datasets.VisionDataset, num_samples: int = 20, cols: int = 4) -> None:
    """ Shows samples from the dataset """
    plt.figure(figsize=(15,15))
    img: torch.Tensor
    for i, img in enumerate(dataset): # type: ignore
        if i == num_samples:
            break
        plt.subplot(num_samples//cols + 1, cols, i + 1)
        plt.imshow(img[0])

def forward(
    image: TensorType["batch", "channel", "x_size", "y_size", float],
    precomputed_variables: PreComputedVariables,
    stepsize: int
    ) -> Tuple[List[int], List[TensorType["batch_1", "channel", "x_size", "y_size", float]], List[TensorType["batch_1", "channel", "x_size", "y_size", float]]]:
    # Get image
    image = image[:1, :]
    times = precomputed_variables.times

    # Plot image for different values of T
    X_t = []
    epsilon_t = []
    t_values = list(range(0, times, stepsize))
    for idx in t_values:
        t = torch.Tensor([idx]).type(torch.int64)
        x_t, eps_t = precomputed_variables.forward_sample(image, t)
        X_t.append(x_t)
        epsilon_t.append(eps_t)
    return t_values, X_t, epsilon_t

def plot_forward(image: TensorType["batch", "channel", "x_size", "y_size", float], precomputed_variables: PreComputedVariables, num_images: int = 10) -> None:
    # Get image
    image = image[:1, :]

    times = precomputed_variables.times
    stepsize = times//num_images
    ts, xs, eps = forward(image, precomputed_variables, stepsize)

    # Plot image for different values of T
    fig, axes = plt.subplots(1, num_images, figsize=(15,15))

    for t, x_t, eps_t, ax in zip(ts, xs, eps, axes):
        ax.set_title(f"T = {t}")
        ax.set_axis_off()
        # Take one sample if we have a batch
        plot_image = x_t
        if len(x_t.shape) == 4:
            plot_image = x_t[0, :, :, :]
        ax.imshow(plot_image[0])
    plt.show()

def plot_sample_image(
    model: Unet,
    precomputed_variables: PreComputedVariables,
    shape: Tuple[int,...],
    device: DEVICES,
    time_max: int,
    n_samples: int = 1,
    num_images: Optional[int] = None,
    plot_t_values: Optional[List[int]] = None,
    figsize: Optional[Tuple[int, int]] = None
    ) -> None:
    if num_images is None and plot_t_values is None:
        raise ValueError("Either num_images or plot_t_values should not be None")
    # Sample noise
    image = torch.randn((n_samples, *shape), device=device)
    plot_backwards(image, model, precomputed_variables, device, time_max, num_images, plot_t_values, figsize)

@torch.no_grad()
def plot_backwards(
    image: TensorType["batch", "channel", "x_size", "y_size", float],
    model: Unet,
    precomputed_variables: PreComputedVariables,
    device: DEVICES,
    time_max: int,
    num_images: Optional[int] = None,
    plot_t_values: Optional[List[int]] = None,
    figsize: Optional[Tuple[int, int]] = None
    ) -> None:
    if num_images is None and plot_t_values is None:
        raise ValueError("Either num_images or plot_t_values should not be None")

    # Backwards operation
    images, ts = backward(model, precomputed_variables, image, device, time_max)

    # plot images
    n_images = len(plot_t_values) if num_images is None else num_images
    figsize = (15,15) if figsize is None else figsize
    fig, axes = plt.subplots(image.shape[0], n_images, figsize=figsize)
    axes = axes.reshape(image.shape[0],-1)

    for i in range(image.shape[0]):
        if num_images is not None:
            stepsize = time_max//num_images
            image_idx = num_images - 1
        else:
            image_idx = len(plot_t_values) - 1
        for img, t in zip(images, ts):
            if (
                (num_images is not None and t % stepsize == 0)
                or (plot_t_values is not None and t in plot_t_values)
            ) and (image_idx >= 0):
                axes[i, image_idx].set_title(f"T = {t}")
                axes[i, image_idx].set_axis_off()

                plot_image = img
                if len(img.shape) == 4:
                    plot_image = img[i, :, :, :]
                axes[i, image_idx].imshow(plot_image[0].detach().cpu())
                image_idx -= 1
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def backward(
    model: Unet,
    precomputed_variables: PreComputedVariables,
    image: TensorType["batch", "channel", "x_size", "y_size", float],
    device: DEVICES,
    time_max: int
    ) -> Tuple[List[TensorType["batch_1", "channel", "x_size", "y_size", float]], List[int]]:

    # Sample and plot images
    images = [image]
    ts = list(reversed(range(0, time_max)))
    for i in ts:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        image = precomputed_variables.generate_sample(model, image, t)
        images.append(image)
    return images, [time_max] + ts
# pylint: disable=no-member



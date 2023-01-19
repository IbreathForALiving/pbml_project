import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from . import DEVICES
from .diffusion import PreComputedVariables, plot_sample_image
from .model import Unet, compute_loss
import matplotlib.pyplot as plt

def train(
    model: Unet,
    dataloader: DataLoader,
    device: DEVICES,
    optimizer: Optimizer,
    precomputed_variables: PreComputedVariables,
    epochs: int,
    ) -> bool:
    batch_size = dataloader.batch_size
    if batch_size is None:
        return False
    for epoch in range(epochs):
        for step, (images, labels) in enumerate(dataloader):
            optimizer.zero_grad()

            # Sample t, noise
            t: torch.Tensor = precomputed_variables.time_sample(batch_size, device)
            x_noisy, noise = precomputed_variables.forward_sample(images, t, device)

            # Predict noise
            noise_pred = model(x_noisy, t)

            # Compute loss
            loss: torch.Tensor = compute_loss(noise, noise_pred)
            loss.backward()
            optimizer.step()
            if step == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
            # if epoch % 5 == 0 and step == 0:
                # print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                # plot_sample_image(model, precomputed_variables, tuple(images[0].shape), device, 10)
    return True
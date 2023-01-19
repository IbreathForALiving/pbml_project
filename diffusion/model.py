""" Module for implementing a UNET model."""
import torch
from torch import nn
import torch.nn.functional as F
import math
from torchtyping import TensorType

def compute_loss(noise: TensorType[float], noise_pred: TensorType[float]) -> TensorType[float]:
    """ Computes the loss of a diffusion model"""
    return F.mse_loss(noise, noise_pred)

class Block(nn.Module):
    """ UNet block for either up or down sampling"""
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, up=False) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.time_emb_dim = time_emb_dim
        self.up = up
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(
        self,
        x: TensorType["batch", "in_channel", "x_size_in", "y_size_in", float],
        t: TensorType["batch", "time", float]
        ) -> TensorType["batch", "out_channel", "x_size_out", "y_size_out", float]:
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal encoding of the time position.
    See https://kazemnejad.com/blog/transformer_architecture_positional_encoding/ for details.
    This block is very typically used in position encoding.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: TensorType["batch", int]) -> TensorType["batch", "time", float]:
        """ Computes the positional embeddings"""
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Unet(nn.Module):
    """
    Simplified U-Net

    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = 1
        down_channels = [64, 128, 256]
        up_channels = down_channels[::-1]
        out_dim = 1 
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList(
            [Block(down_channels[i], down_channels[i+1], time_emb_dim) for i in range(len(down_channels)-1)]
        )
        # Upsample
        self.ups = nn.ModuleList(
            [Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True) for i in range(len(up_channels)-1)]
        )

        self.output = nn.Conv2d(up_channels[-1], image_channels, out_dim)

    def forward(self, x: TensorType["batch", "channel", "x_size", "y_size", float], timestep: TensorType["batch", torch.long]):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        return self.output(x)
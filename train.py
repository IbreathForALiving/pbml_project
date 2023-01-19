from diffusion.data import load_mnist_dataset
from diffusion.diffusion import PreComputedVariables, plot_forward
from diffusion.model import Unet
from diffusion.training import train
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam

if __name__ == '__main__':
    # Constants
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device = {DEVICE}")
    T = 30
    BATCH_SIZE = 128
    EPOCHS = 100

    # Load data and setup variables
    data = load_mnist_dataset(Path('.'), download=True)
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    precomputed_variables = PreComputedVariables(T)

    # plot_forward(dataloader, precomputed_variables)
    model = Unet().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=0.001)
    for i in range(10):
        train(model, dataloader, DEVICE, optimizer, precomputed_variables, EPOCHS)
        torch.save(model.state_dict(), f"MODEL{EPOCHS*i}.pt")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class conv_encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super(conv_encoder, self).__init__()
        self.name = "conv_decoder"
        self.in_dim = input_dim
        self.latent_dim = latent_dim

        # convolution & maxpool: dim_out = floor((dim_in+2*padding-dilation*(kernel_size-1)-1)/stride + 1)
        # default: padding=0, dilation=1, stride=1
        self.conv_pool = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3)), # 26*26
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(3,3)) # default stride=kernel_size: 8*8
        )
        dim = int(((np.sqrt(self.in_dim) - 3 + 1) - 3)/3 + 1) # (in_dim-3 + 1) new dim after convolution
        self.hidden = nn.Linear(dim*dim, 400)

        self.mu = nn.Linear(400, self.latent_dim)
        self.sigma = nn.Linear(400, self.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = F.leaky_relu(self.hidden(x)) # F.elu(self.hidden(x))
        return self.mu(x), self.sigma(x)

class conv_decoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(conv_decoder, self).__init__()
        self.name = "conv_decoder"
        self.in_dim = input_dim
        self.out_dim = output_dim

        self.layers = nn.Sequential(
            nn.Linear(self.in_dim, 400),
            nn.LeakyReLU(), # nn.ELU(),
            nn.Linear(400, self.out_dim),
            nn.Sigmoid() # nn.LeakyReLU()
        )
        """self.rescaler = nn.Sequential(
            nn.ConvTranspose2d(1, 1, 3),
            nn.Sigmoid()
        )"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        out = int(np.sqrt(self.out_dim))
        return x.view(-1, 1, out, out)

# Convolutional VAE
class Conv_VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, output_dim: int) -> None:
        super(Conv_VAE, self).__init__()
        self.name = "Conv_VAE"
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.latent_dim = latent_dim

        self.encoder = conv_encoder(self.in_dim, self.latent_dim)
        self.decoder = conv_decoder(self.latent_dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, logsigma = self.encoder(x)

        # reparametrize
        std = torch.exp(logsigma)
        eps = torch.randn_like(std) # filled with random numbers from a normal distribution with mean 0 and variance 1
        # mu + eps*std follows N(mu, std^2 I) (mean 0, variance 1)
        # because eps follows N(0, I)
        tmp = mu + std * eps
        return self.decoder(tmp), std.pow(2), mu



def train(epoch, model, train_loader):
    l = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        model.train()
        data = data.to(device)
        reconstr, mu, var = model(data)
        loss = loss_function(reconstr, data, mu, var)
        l += loss.item()
        loss.backward()
        radam.step()
        radam.zero_grad()

    return l / len(train_loader.dataset)

def test(epoch, model, test_loader):
    l = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            reconstr, mu, var = model(data)
            loss = loss_function(reconstr, data, mu, var)
            l += loss.item()

    return l / len(test_loader.dataset)

"""
# training exemple

conv_vae = Conv_VAE(28*28, 2, 28*28).to(device)
print(conv_vae)

radam = optim.RAdam(conv_vae.parameters(), lr=1e-3)

N = 50

train_l, test_l = [], []

for epoch in tqdm(range(N)):
    train_l.append(train(epoch, conv_vae, train_loader))
    test_l.append(test(epoch, conv_vae, test_loader))

plt.plot(list(range(1, N+1)), train_l, label="train loss")
plt.plot(list(range(1, N+1)), test_l, label="test loss")
plt.legend()
plt.show()

torch.save(conv_vae.state_dict(), 'conv_vae.pt')
"""

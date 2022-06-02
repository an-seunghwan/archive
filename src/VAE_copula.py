#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#%%
import statsmodels
from statsmodels.distributions.copula.api import ClaytonCopula

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt
#%%
import sys
import subprocess
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("../wandb_api.txt", "r") as f:
        key = f.readlines()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

wandb.init(
    project="RafterNet", 
    entity="anseunghwan",
    tags=["clayton"],
)
#%%
import argparse
def get_args(debug):
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--seed', type=int, default=1, 
                        help='seed for repeatable results')
    parser.add_argument('--n', type=int, default=50000, 
                        help='the number of dataset')
    parser.add_argument('--data_dim', type=int, default=2, 
                        help='dimension of copula')
    parser.add_argument('--tau', type=float, default=0.5, 
                        help='Kendalls tau of copula')
    
    parser.add_argument('--train_iter', type=int, default=50, 
                        help='the number of training iteration')
    
    parser.add_argument('--latent_dim', type=int, default=200, 
                        help='dimension of latent variable')
    parser.add_argument('--num_layers', type=int, default=1, 
                        help='number of neural network layers')
    parser.add_argument('--hidden_dim', type=int, default=100, 
                        help='number of neural network layers')

    if debug:
        return parser.parse_args(args=[])
    else:    
        return parser.parse_args()
#%%
class VAE(nn.Module):
    def __init__(self,
                data_dim,
                latent_dim,
                num_layers,
                hidden_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Build Encoder
        encoder = []
        for i in range(num_layers):
            in_dim = hidden_dim
            out_dim = hidden_dim
            if i == 0: in_dim = data_dim
            encoder.append(nn.Linear(in_dim, out_dim))
            encoder.append(nn.ReLU())
        self.encoder = nn.ModuleList(encoder)

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        # Build Decoder
        decoder = []
        for i in range(num_layers):
            in_dim = hidden_dim
            out_dim = hidden_dim
            if i == 0: in_dim = latent_dim
            decoder.append(nn.Linear(in_dim, out_dim))
            decoder.append(nn.ReLU())
        decoder.append(nn.Linear(hidden_dim, data_dim))
        self.decoder = nn.ModuleList(decoder)

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network and returns the latent codes.
        """
        h = input
        for e in self.encoder:
            h = e(h)
        mean = self.mean(h)
        logvar = self.logvar(h)
        return [mean, logvar]

    def reparameterize(self, mean, logvar):
        """
        Reparameterization trick to sample from N(mean, var) from N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mean + epsilon * std

    def forward(self, input):
        h = input
        for e in self.encoder:
            h = e(h)
        mean = self.mean(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mean, logvar)
        h = z
        for d in self.decoder:
            h = d(h)
        return  [h, z, mean, logvar]

    def sample(self, num_samples):
        """
        Samples from the latent space and return the corresponding data space map.
        """
        z = torch.randn(num_samples, self.latent_dim)
        for d in self.decoder:
            z = d(z)
        return z

    def generate(self, x):
        """
        Given an input image x, returns the reconstructed data
        """
        return self.forward(x)[0]
#%%
def main():
    config = vars(get_args(debug=True)) # default configuration
    wandb.config.update(config)
    
    n = config["n"]
    data_dim = config["data_dim"]
    tau = config["tau"]
    cop = ClaytonCopula(theta=tau, k_dim=data_dim)
    data = cop.rvs(n, random_state=config["seed"])
    data = torch.FloatTensor(data)
    
    model = VAE(
        data_dim=data_dim, 
        latent_dim=config["latent_dim"], 
        num_layers=config["num_layers"], 
        hidden_dim=config["hidden_dim"], 
    )
    print(model)
    
    def loss_function(data, xhat, mean, logvar):
        beta = 1.
        recon_loss = 0.5 * torch.mean(torch.sum(torch.pow(xhat - data, 2), axis=1))
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean ** 2 - logvar.exp(), dim = 1), dim = 0)
        loss = recon_loss + beta * kl_loss
        return {'loss': loss, 'recon_loss':recon_loss, 'KLD':kl_loss}
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for iteration in range(config["train_iter"]):
        optimizer.zero_grad()
        xhat, z, mean, logvar = model(data)
        loss_ = loss_function(data, xhat, mean, logvar)
        loss = loss_['loss']
        loss.backward()
        optimizer.step()
        
        print_input = "[iteration {:03d}]".format(iteration)
        print_input += ''.join([', {}: {:.4f}'.format(x, y.detach().item()) for x, y in loss_.items()])
        print(print_input)
        
        wandb.log({x : y.detach().item() for x, y in loss_.items()})
    
    """
    http://webdoc.sub.gwdg.de/ebook/serien/e/uio_statistical_rr/05-07.pdf
    """
    n_gen = 1000
    data_gen = model.sample(n_gen)
    data_gen = data_gen.detach().numpy()
    S = 0
    for i in range(n_gen):
        a = ((data_gen <= data_gen[i, :]).sum(axis=1) == 2).astype(float).sum() / (n_gen + 1)
        b = cop.cdf(data_gen[i, :])[0]
        S += (a - b) ** 2
    S = S / n_gen
    wandb.log(S)
    
    wandb.run.finish()
#%%
if __name__ == "__main__":
    main()
#%%
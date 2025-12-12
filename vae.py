import torch 
import torch
import torch.nn as nn

#TODO: Hyperparams

def conv_block(in_channels, out_channels,output_padding=0,up=True):
    ConvLayer = nn.ConvTranspose2d if up else nn.Conv2d
    return nn.Sequential(
        ConvLayer(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU()
    )
    
class VAE(nn.Module):
    def __init__(self, channels, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(channels, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            nn.Flatten()
        )
        encoder_output_dim= 128 * 3 * 3
        self.fc_mean = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, encoder_output_dim)
        
        self.decoder = nn.Sequential(
            conv_block(4, 32, output_padding=1,Up=False),
            conv_block(32, 64,Up=False),
            nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1,Up=False),
            nn.Sigmoid()
        )
    def loss(x_hat,x,mean,logvar,beta):
        nll = nn.MSELoss()(x_hat,x)
        kld =  -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return nll + beta*kld
    
    def encode(self, x):
        x = self.encoder(x)
        return self.fc_mean(x), self.fc_logvar(x)

    def reparam(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

    def decode(self, z):
        z = self.fc_decoder(z)
        z = z.view(z.size(0), 128, 3, 3)
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparametrize(mean, logvar)
        out = self.decode(z)
        return out, mean, logvar
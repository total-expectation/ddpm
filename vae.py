import torch
import torch.nn as nn

def conv_block(in_channels, out_channels, up=False):
    ConvLayer = nn.ConvTranspose2d if up else nn.Conv2d

    if up:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, 
                out_channels, 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                output_padding=1

            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

class VAE(nn.Module):
    def __init__(self, channels, latent_dim):
        super().__init__()
        down_channels = (32, 64, 128)
        up_channels = down_channels[::-1]

        conv_encoder = nn.Sequential(
            conv_block(channels, down_channels[0]), 
            conv_block(down_channels[0], down_channels[1]),
            conv_block(down_channels[1], down_channels[2]),
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, 28, 28) 
            
            dummy_4d_out = conv_encoder(dummy_input)
            self.enc_shape = dummy_4d_out.shape[1:] 
            
            dummy_flat_out = dummy_4d_out.flatten(1)
            
            encoder_output_dim = dummy_flat_out.size(1) 

        self.encoder = nn.Sequential(
            conv_encoder,
            nn.Flatten()
        )

        self.fc_mean = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, encoder_output_dim)

        self.decoder = nn.Sequential(
            conv_block(down_channels[2], up_channels[0], up=True),
            conv_block(up_channels[0], up_channels[1], up=True),
            nn.ConvTranspose2d(up_channels[1], channels, kernel_size=4, stride=2, padding=2)
        )

    def loss(self, x_hat, x, mean, logvar, beta):
        nll = nn.MSELoss()(x_hat, x)
        kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return nll + beta * kld

    def encode(self, x):
        x = self.encoder(x)
        return self.fc_mean(x), self.fc_logvar(x)

    def reparam(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

    def decode(self, z):
        z = self.fc_decoder(z)
        z = z.view(z.size(0), *self.enc_shape)
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparam(mean, logvar)
        out = self.decode(z)
        return out, mean, logvar
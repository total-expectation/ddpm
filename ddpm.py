import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from attention import MultiHeadAttention
from torchvision import datasets
from torchvision.transforms import transforms
import matplotlib as plt
print(torch.__version__)
print(plt.__version__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Hyperparams
BATCH_SIZE = 128  # according to the paper

# CIFAR10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.RandomHorizontalFlip()  # p=0.5 by default
])

train = datasets.CIFAR10("../data", train=True, download=True, transform=transform)
test = datasets.CIFAR10("../data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

    
# architecture
class ResNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, time_embedding_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.time_emb_proj = nn.Linear(time_embedding_dim, out_channel)  # TODO: ??? What is the right dimensions??
        self.group_norm1 = nn.GroupNorm(32, out_channel)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.group_norm2 = nn.GroupNorm(32, out_channel)
        self.out=nn.Conv2d(in_channel,out_channel,kernel_size=1)

    def forward(self, x,time_emb):
        #TODO make sure x is same dimension as 
        #torch.Size([128, 128, 28, 28])
        
        h = self.conv1(x) 
        temb_proj = self.time_emb_proj(time_emb).unsqueeze(-1).unsqueeze(-1)
        
        h += temb_proj
        h = self.gelu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.group_norm2(h)

        x = self.out(x)
        return h + x
        
class AttentionBlock(ResNetBlock):
    
    def __init__(self, in_channel, out_channel, time_embedding_dim, dropout=0.1):
        super().__init__(in_channel, out_channel, time_embedding_dim, dropout)
        self.attention=MultiHeadAttention(in_dim=out_channel,out_dim=out_channel)
        
    def forward(self, x,time_emb):
        h=super().forward(x,time_emb)
        h=self.attention(h)
        return h
        
class SPE(nn.Module):
    """
    Create sinusoidal position embeddings at differnet time steps
    """
    def __init__(self,dim):
        super().__init__()
        self.dim=dim
        
    #TODO: we start with the not learned sinusoidal embedding, later can add MLP to learn it
    def forward(self,time):
        import math
        device = time.device
        half_dim = self.dim//2
        emb = math.log(10000) / (half_dim-1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -emb)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Unet(nn.Module):
    """
    Time Conditioned UNet
    """
    def __init__(self,in_channels=1):
        super().__init__()
        down_channels=(128,128,256,256,512,512)
        up_channels=down_channels[::-1]
        output_dimensions= in_channels
        time_embedding_dim=256
        
        self.proj=nn.Conv2d(in_channels,down_channels[0],kernel_size=3,padding=1)
        self.mlp=nn.Sequential(
            SPE(time_embedding_dim),
            nn.Linear(time_embedding_dim,time_embedding_dim),
            nn.SiLU()
        )
        
        self.down=nn.ModuleList([
            ResNetBlock(down_channels[0],down_channels[1],time_embedding_dim),
            ResNetBlock(down_channels[1],down_channels[2],time_embedding_dim),
            ResNetBlock(down_channels[2],down_channels[3],time_embedding_dim),
            AttentionBlock(down_channels[3],down_channels[4],time_embedding_dim),
            ResNetBlock(down_channels[4],down_channels[5],time_embedding_dim)
        ])

        self.bottleneck=nn.ModuleList([
            ResNetBlock(down_channels[5],down_channels[5],time_embedding_dim),
            AttentionBlock(down_channels[5],up_channels[0],time_embedding_dim)
            #ResNetBlock(down_channels[-1],up_channels[0])
        ])

        self.up=nn.ModuleList([
            ResNetBlock(up_channels[0],up_channels[1],time_embedding_dim),
            AttentionBlock(up_channels[1],up_channels[2],time_embedding_dim),
            ResNetBlock(up_channels[2],up_channels[3],time_embedding_dim),
            ResNetBlock(up_channels[3],up_channels[4],time_embedding_dim),
            ResNetBlock(up_channels[4],up_channels[5],time_embedding_dim)
        ])
        #project back up to original size
        self.output = nn.Conv2d(up_channels[-1], in_channels,kernel_size=3,padding=1)
    
    def forward(self,x,t):
        # Get time embedding
        time_embedding=self.mlp(t)
        x=self.proj(x)
        
        skip_connection = []
        for block in self.down:
            skip_connection.append(x)
            x=block(x,time_embedding)

        skip_connection_bottom = x
        for block in self.bottleneck:
            x=block(x,time_embedding)
        x += skip_connection_bottom

        for block in self.up:
            x=block(x,time_embedding)
            x = x + skip_connection.pop()
            
        return self.output(x)
        

class DDPM(nn.Module):
    def __init__(self, T=1000):
        super().__init__()
        self.T = T
        self.epsilon_model=Unet(1)
        #beta
        #alpha
        #alphabar
        #beta_tilde
        #sigma
        self.beta = torch.linspace(1e-4, 2e-2, T).to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha,dim=0)
        self.alpha_bar_0=torch.tensor([1.0],device=self.alpha_bar.device)
        self.alpha_bar=torch.cat([self.alpha_bar_0,self.alpha_bar])
        #self.register_buffer('alpha_bar', torch.cat([self.alpha_bar_0, self.alpha_bar]))
        self.beta_tilde_1_to_T = (1.0 - self.alpha_bar[:-1]) / (1.0 - self.alpha_bar[1:]) * self.beta
        self.beta_tilde_0 = torch.tensor([0.0], device=self.alpha_bar.device)
        self.beta_tilde = torch.cat([self.beta_tilde_0, self.beta_tilde_1_to_T])
        
    def forward(self,x):
        """
        Alg 1 from DDPM Paper
        1. Sample random time step
        2. sample random noise 
        3. noise input to create x_t
        4. predict noise using unet
        5. return loss between predicted and actual noise
        """
        bs = x.shape[0]
        t = torch.randint(low=0, high=self.T, size=(bs,), device=x.device)
        noise = torch.randn(x.shape, device=x.device)
        noisy_img = self.diffusion_kernel(x, t, noise)
        pred_noise=self.epsilon_model(noisy_img, t)
        
        return pred_noise, noise

    def diffusion_kernel(self, x, t, noise):
        # Get the alpha values for the specific time steps
        alpha_t = self.alpha_bar[t]
        
        # Reshape alpha_t to (Batch, 1, 1, 1) so it can broadcast against the image (Batch, C, H, W)
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        
        return torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise

        #return torch.sqrt(self.alpha_bar[t]) * x + torch.sqrt(1 - self.alpha_bar[t]) * noise

    def sample(self,num_samples,size=(1,28,28),t=0):
        """
        Sample image from random noise.
        Alg 2 from DDPM paper.
        1. Starts from pure noise.
        2. Denoises to generate new samples.
        """
        #generate random sample for t=T, X_T= N(0,1)
        x_T = torch.random(num_samples, *size).to(device)
        #step time backwards
        x_u=x_T
        for u in range (self.T,t,-1):
            z=torch.randn_like(x_u) if u > t+1 else 0
            epsilon = self.epsilon_model(x_u,u) # noise prediction network
            #need values from different time steps
            sigma_t = (1 - self.alpha_bar[u-1])/(1 - self.alpha_bar[u]) * self.beta[u]
            x_u_minus_1 = (1 / torch.sqrt(self.alpha[u]) *( x_u - (1 - self.alpha[u]) / torch.sqrt(1 - self.alpha_bar[u]) * epsilon) + sigma_t*z )
            x_u=x_u_minus_1
        return x_u

    def loss(pred_noise, true_noise):
        return F.mse_loss(pred_noise, true_noise)
        
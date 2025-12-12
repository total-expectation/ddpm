import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
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

#DDPM Schedules

# DDPM needs schedules to control how noise is added and removed.
def schedules():
        
    return 
    
# architecture
class ResNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, time_embedding_dim, dropout=0.1):
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=0)
        self.time_emb_proj = nn.Linear(time_embedding_dim, out_channel)  # TODO: ??? What is the right dimensions??
        self.group_norm1 = nn.GroupNorm(out_channel, out_channel)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=0),
        self.group_norm2 = nn.GroupNorm(out_channel, out_channel)
        

    def forward(self, x,time_emb):
        #TODO make sure x is same dimension as 
        h = self.conv1(x)
        temb_proj = self.time_emb_proj(time_emb)
        h += temb_proj
        h = self.gelu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.group_norm2(h)

        return h + x
        

class Bottleneck(nn.Module):
    def __init__(self, in_dim):
        pass

class SPE(nn.Module):
    """
    Create sinusoidal position embeddings at differnet time steps
    """
    def __init__(self,dim):
        super().__init__()
        self.dim=dim
        
    #TODO: we start with the not learned sinusoidal embedding, later can add MLP to learn it
    def forwards(self,time):
        device = time.device
        half_dim = self.dim//2
        emb = torch.log(10000) / (half_dim-1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -emb)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Unet(nn.Module):
    """""
    Time Conditioned UNet
    """"
    def __init__(self,in_channels=1):
        super().__init__()
        down_channels=(128,128,256,256,512,512)
        up_channels=(512,512,256,256,128,128)
        output_dimensions= in_channels
        time_embedding_dim=256
        
        self.mlp=nn.Sequential(
            SPE(time_embedding_dim),
            nn.Linear(time_embedding_dim,time_embedding_dim),
            nn.SiLU()
        )
        self.proj=nn.conv2d(in_channels,)
        self.up=nn.ModuleList([
     
        ])
        self.bottleneck=nn.Module([
            
        ])
        self.down=nn.ModuleList([
            
        ])
        
    
    
    def forward(x,t):
        # Get time embedding
        time_embedding=self.mlp(t)
        # Covolution
        
        # Skip connections
    
        # Down sampling
        
        # Bottlneck 
        
        # Sampling
        
class DDPM:
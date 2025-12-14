import torch
import torch.nn as nn
import torch.nn.functional as F

# x - embedding
# Q,K,V matrices
# x_

    
class SelfAttention(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.Q = nn.Linear(in_dim, out_dim)
        self.K = nn.Linear(in_dim, out_dim)
        self.V = nn.Linear(in_dim, out_dim)
        self.group_norm = nn.GroupNorm(32, in_dim)
        
        
    def compute_qkv(self, X):
        print(X.shape)
        X.transpose(dim0=1, dim1=-1)
        print(X.shape)
        Q_x = self.Q(X)
        K_x = self.K(X)
        V_x = self.V(X)
        
        return Q_x, K_x, V_x
    
    def compute_attention_matrix(self, Q, K):
        d_k = self.out_dim
        scores= Q @ K.transpose(-2,-1) / torch.sqrt(d_k)
        return F.softmax(scores,dim=-1)

    def forward(self, X):
        X = self.group_norm(X) #TODO: not sure about the output dimen, need to experiment??
        Q, K, V = self.compute_qkv(X)
        scores = self.compute_attention_matrix(Q, K)
        return scores @ V
    
class MultiHeadAttention(nn.Module):
    def __init__(self, c_dim, in_dim, heads):
        self.C = nn.Linear(c_dim)
        self.heads = heads
        self.multi_attention = nn.ModuleList(
            SelfAttention(in_dim / self.heads, in_dim / self.heads, in_dim / self.heads) for _ in range(self.heads)
        )

    def forward(self, X):
        return self.C(torch.cat([self_attn_block(X) for self_attn_block in self.multi_attention()]))  # TODO: not sure about concat dimension
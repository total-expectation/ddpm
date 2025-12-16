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
        if X.dim() == 4:
            B, C, H, W = X.shape
            # move channels to last dim and flatten spatial dims -> (B, L, C)
            X = X.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        elif X.dim() == 3:
            # assume already (B, L, C)
            pass
        else:
            raise ValueError(f"Unsupported input shape for SelfAttention: {X.shape}")

        Q_x = self.Q(X)
        K_x = self.K(X)
        V_x = self.V(X)
        
        return Q_x, K_x, V_x
    
    def compute_attention_matrix(self, Q, K):
        d_k = self.out_dim
        scores= Q @ K.transpose(-2,-1) / (d_k ** 0.5)
        return F.softmax(scores,dim=-1)

    def forward(self, X):
        X = self.group_norm(X) #TODO: not sure about the output dimen, need to experiment??
        Q, K, V = self.compute_qkv(X)
        scores = self.compute_attention_matrix(Q, K)
        return scores @ V
    
class MultiHeadAttention(nn.Module):
    # def __init__(self, c_dim, in_dim, heads):
    #     self.C = nn.Linear(c_dim)
    #     self.heads = heads
    #     self.multi_attention = nn.ModuleList(
    #         SelfAttention(in_dim / self.heads, in_dim / self.heads, in_dim / self.heads) for _ in range(self.heads)
    #     )

    # def forward(self, X):
    #     return self.C(torch.cat([self_attn_block(X) for self_attn_block in self.multi_attention()]))  # TODO: not sure about concat dimension
    def __init__(self, in_dim, out_dim=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim else in_dim
        
        self.group_norm = nn.GroupNorm(32, in_dim)
        # Use PyTorch's built-in MultiheadAttention
        self.attention = nn.MultiheadAttention(in_dim, num_heads=8, batch_first=True)
        self.to_out = nn.Conv2d(in_dim, in_dim, kernel_size=1)
    
    def forward(self, X):
        # X shape: (B, C, H, W)
        B, C, H, W = X.shape
        X = self.group_norm(X)
        
        # Reshape to (B, L, C) for attention
        X_flat = X.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        
        # Apply attention
        attn_out, _ = self.attention(X_flat, X_flat, X_flat)
        
        # Reshape back to (B, C, H, W) BEFORE self.to_out
        attn_out = attn_out.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        
        # Now apply Conv2d
        out = self.to_out(attn_out)
        
        return out
import torch
import torch.nn as nn
from requests.api import head


class Attention(nn.Module):
    def __init__(self, init_dim, heads=16, head_dim=64):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim

        # normalize our tensor
        self.norm = nn.LayerNorm(init_dim)
        # now each head gets 64 so 16*64=1024 total per token
        # so 192 -> 1024 except 3 because K and Q and V
        self.toQKV = nn.Linear(init_dim, 3 * heads * head_dim)
        self.backToInitDim = nn.Linear(init_dim, heads * head_dim)

    def forward(self, x: torch.Tensor):
        # x is shaped somthing like [B, T, 192]
        x = self.norm(x)
        # create the Q, K, V
        # shape is now [B, T, 1024x3]
        qkv = self.toQKV(x)
        # chunk it into 3 on the last dimension
        # now each is [B, T, H*D]
        q, k, v = qkv.chunk(3, -1)

        B, T, _ = x.shape
        # we want q,k,v [B, T, H*D] -> [B, T, H, D] -> [B, H, T, D]
        q = q.reshape(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.heads, self.head_dim).transpose(1, 2)

        # now its time for classic Attention
        # now to do the q x k is simple:
        # [B, H, T, D] x [B, H, D, T]
        scores = q @ k.transpose(-2, -1)
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v

        # now get it back to what we want
        out = out.transpose(1, 2)  # [B, T, H, D]
        out = out.reshape(B, T, self.heads * self.head_dim)  # [B, T, H*D]
        out = self.backToInitDim(out)
        return out

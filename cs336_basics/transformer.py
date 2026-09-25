import torch 
import torch.nn as nn
import math
from einops import einsum 

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        
        shape = (out_features, in_features)
        self.weight = nn.Parameter(torch.zeros(shape, dtype=dtype, device=device))

        sigma = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=sigma, a=-3*sigma, b=3*sigma)

    def forward(self, x):
        x = einsum(x, self.weight, '... d_in, d_out d_in -> ... d_out')
        return x
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        shape = (num_embeddings, embedding_dim)
        self.weight = nn.Parameter(torch.zeros(shape, dtype=dtype, device=device))

        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1,  a=-3, b=3)

    def forward(self, token_ids):
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))


    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean_square = torch.mean(torch.square(x), dim=-1, keepdim=True)
        rms = torch.sqrt(mean_square + self.eps)
        rmsnorm = (x * self.weight) / (rms)

        return rmsnorm.to(in_dtype)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()

        if(d_ff is None):
            d_ff = (8/3) * d_model
            d_ff = 64 * round(d_ff / 64)

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x):
        z = self.w1(x)
        branch1 = z * torch.sigmoid(z)
        branch2 = self.w3(x)
        
        swiglu = self.w2(branch1 * branch2)

        return swiglu

class RoPE(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()

        indices = torch.arange(start=0, end=d_k, step=2, device=device) / d_k
        inv_freq = 1 / (theta**indices)

        positions = torch.arange(start=0, end=max_seq_len, device=device)
        positions = positions.unsqueeze(-1)

        angles = positions * inv_freq
        sine_values = torch.sin(angles)
        cosine_values = torch.cos(angles)
        self.register_buffer("sine_values", sine_values, persistent=False)
        self.register_buffer("cosine_values", cosine_values, persistent=False)

    def forward(self, x, token_positions):
        even_indexed = x[..., 0::2]
        odd_indexed = x[..., 1::2]
        sin = self.sine_values[token_positions]
        cos = self.cosine_values[token_positions]

        rotated_even = (even_indexed * cos) - (odd_indexed * sin)
        rotated_odd = (even_indexed * sin) + (odd_indexed * cos)

        stacked = torch.stack((rotated_even, rotated_odd), dim=-1)
        return stacked.flatten(-2)
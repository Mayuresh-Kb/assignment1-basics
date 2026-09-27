import torch 
import torch.nn as nn
import math
from einops import einsum, rearrange

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
    
def softmax(x, dim):
    x_max = torch.max(x, dim=dim, keepdim=True)
    x_stable = x - x_max.values
    exp_x = torch.exp(x_stable)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return (exp_x / sum_exp)

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    dot_product = einsum(Q,K, '... n d_k, ... m d_k -> ... n m')
    scaled_scores = dot_product / math.sqrt(d_k)

    if(mask is not None):
        scaled_scores = scaled_scores.masked_fill(~mask, -torch.inf)

    attention_weights = softmax(scaled_scores, dim=-1)
    return einsum(attention_weights,V, '... n m, ... m d_v -> ... n d_v')

class Causal_multi_head_self_attention(nn.Module):
    def __init__(self, d_model, num_heads, theta=None, max_seq_len=None):
        super().__init__()

        self.num_heads = num_heads
        self.d_k = (d_model // self.num_heads)

        self.wq = Linear(in_features=d_model, out_features=d_model)
        self.wk = Linear(in_features=d_model, out_features=d_model)
        self.wv = Linear(in_features=d_model, out_features=d_model)
        self.wo = Linear(in_features=d_model, out_features=d_model)

        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta, self.d_k, max_seq_len)
        else:
            self.rope = None

    def forward(self, x, token_positions=None):            
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = rearrange(q, '... seq (h d_k) -> ... h seq d_k', h=self.num_heads, d_k = self.d_k) 
        k = rearrange(k, '... seq (h d_k) -> ... h seq d_k', h=self.num_heads, d_k = self.d_k) 
        v = rearrange(v, '... seq (h d_k) -> ... h seq d_k', h=self.num_heads, d_k = self.d_k)

        if token_positions is not None:
            rope_positions = token_positions.unsqueeze(1)
            q = self.rope(q, rope_positions)
            k = self.rope(k, rope_positions)

        seq_len = x.shape[-2]
        positions = torch.arange(start=0, end=seq_len, step=1, device=x.device)
        query_positions = positions.unsqueeze(-1)
        key_positions = positions.unsqueeze(0)

        causal_mask = key_positions <= query_positions
        o = scaled_dot_product_attention(q, k, v, causal_mask)
        o = rearrange(o, '... h seq d_k -> ... seq (h d_k)')
        return self.wo(o)

class Transformer_block(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta, max_seq_len):
        super().__init__()

        self.norm1 = RMSNorm(d_model)
        self.attention = Causal_multi_head_self_attention(d_model, num_heads, theta, max_seq_len)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x):
        seq_len = x.shape[-2]
        positions = torch.arange(start=0, end=seq_len, device=x.device)
        positions = positions.unsqueeze(0)
        y = x + self.attention(self.norm1(x), positions)
        z = y + self.ffn(self.norm2(y))
        return z 
    
class Transformer_lm(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList()
        for layer in range(num_layers):
            self.layers.append(Transformer_block(d_model=d_model, num_heads=num_heads, d_ff=d_ff, theta=rope_theta, max_seq_len=context_length))

        self.rmsnorm = RMSNorm(d_model)
        self.lmhead = Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, in_indices):
        x = self.token_embeddings(in_indices)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.rmsnorm(x)
        x = self.lmhead(x)
        return x

        

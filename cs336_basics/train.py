import torch 
import torch.nn as nn
import math
from collections.abc import Callable, Iterable
from typing import Optional
from einops import einsum, rearrange

def cross_entropy(logits, targets):
    M = torch.max(logits, dim=-1, keepdim=True)[0]
    shifted_logits = logits - M

    second_term = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))
    targets = targets.unsqueeze(-1)

    first_term = torch.gather(shifted_logits, dim=-1, index=targets)
    first_term = first_term.squeeze(-1)

    loss = -first_term + second_term
    return torch.mean(loss)
    
class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr, betas, eps, weight_decay):
        beta1, beta2 = betas
        defaults = {"lr": lr, "beta1": beta1, "beta2": beta2, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                else:
                    g = p.grad
                    state = self.state[p]
                    if not state:
                        state["t"] = 0
                        state["m"] = torch.zeros_like(p)
                        state["v"] = torch.zeros_like(p)
                    
                    state["t"] = state["t"] + 1
                    t = state["t"]
                    m = state["m"]
                    v = state["v"]
                    alpha = group["lr"]
                    beta1 = group["beta1"]
                    beta2 = group["beta2"]
                    epsilon = group["eps"]
                    weight_decay = group["weight_decay"]

                    alpha_t = alpha * (math.sqrt(1 - beta2**t) / (1 - beta1**t))
                    
                    m = (beta1 * m) + (1 - beta1)*g
                    state["m"] = m
                    v = (beta2 * v) + (1 - beta2)*(g**2)
                    state["v"] = v

                    with torch.no_grad():
                        p.sub_(alpha * weight_decay * p)
                        p.sub_(alpha_t * (m / (torch.sqrt(v) + epsilon)))

def learning_rate_schedule(t, alpha_max, alpha_min, T_w, T_c):
    if t < T_w:
        return (t/T_w) * alpha_max
    elif T_w <= t <= T_c:
        r = (t - T_w) / (T_c - T_w)
        return alpha_min + 0.5 * (1 + math.cos(r*math.pi)) * (alpha_max - alpha_min)
    else:
        return alpha_min
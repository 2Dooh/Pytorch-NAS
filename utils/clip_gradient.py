import torch
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_

def clip_gradient(model: Module, clip_val: int) -> None:
    clip_grad_norm_(model.parameters(), clip_val)

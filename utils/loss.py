import torch
import torch.nn as nn

from torch.nn import Module


class KLD_loss(Module):
    def __init__(self,):
        super().__init__()
        
    def forward(self, x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

        return reproduction_loss + KLD
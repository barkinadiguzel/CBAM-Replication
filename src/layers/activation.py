import torch
import torch.nn as nn

def get_activation(name='relu'):
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f"Activation {name} not supported")

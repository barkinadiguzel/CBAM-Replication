import torch.nn as nn

def get_norm(norm_type='batch', num_features=None):
    if norm_type == 'batch':
        return nn.BatchNorm2d(num_features)
    elif norm_type == 'layer':
        return nn.LayerNorm([num_features, 1, 1])
    else:
        raise ValueError(f"Normalization {norm_type} not supported")

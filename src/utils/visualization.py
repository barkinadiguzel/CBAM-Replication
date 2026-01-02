import matplotlib.pyplot as plt
import torch

def show_feature_map(feature_map, idx=0):
    fmap = feature_map[idx].detach().cpu()
    n_channels = fmap.shape[0]
    plt.figure(figsize=(12, 8))
    for i in range(min(n_channels, 16)):
        plt.subplot(4,4,i+1)
        plt.imshow(fmap[i], cmap='viridis')
        plt.axis('off')
    plt.show()

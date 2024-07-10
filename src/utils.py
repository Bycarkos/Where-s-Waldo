import torch
import torch.nn as nn








def contrastive_loss(x1, x2, label, margin: float = 0.2):
    """
    Computes Contrastive Loss
    """

    dist = torch.nn.functional.pairwise_distance(x1, x2)

    loss = (1 - label) * torch.pow(dist, 2) \
        + (label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    loss = torch.mean(loss)

    return loss


def compute_resize(shape, patch_size):
    offset = shape % patch_size
    if offset > (patch_size /2):
        return shape + (patch_size - offset)
    else:
        return shape - offset
    
    
def extract_optimal_shape(min_width:int,  max_height:int, patch_size:int):
    
    min_width_recovered = compute_resize(shape=min_width, patch_size=patch_size)
    max_height_recovered = compute_resize(shape=max_height, patch_size=patch_size)
    
    return min_width_recovered, max_height_recovered
    
    
    

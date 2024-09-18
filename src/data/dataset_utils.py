
import torch
import torch.nn as nn

from  torchtyping import TensorType

import os
import umap
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st 


from typing import *

import pickle






def compute_resize(shape, patch_size):
    offset = shape % patch_size
    if offset > (patch_size /2):
        return shape + (patch_size - offset)
    else:
        return shape - offset
    
    
def extract_optimal_shape(min_width:int,  max_height:int, patch_size:int):
    
    min_width_recovered = compute_resize(shape=min_width, patch_size=patch_size)
    max_height_recovered = compute_resize(shape=max_height, patch_size=patch_size)
    
    return int(min_width_recovered), int(max_height_recovered)
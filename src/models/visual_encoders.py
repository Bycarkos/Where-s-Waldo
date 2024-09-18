import torch.nn as nn
import torch


import math
from hydra.utils import instantiate
from omegaconf import DictConfig

import pdb

device = "cuda" if torch.cuda.is_available else "cpu"

from transformers import TrOCRProcessor

#    min_height=kernel_height, max_width=kernel_width, hidden_channels=64, output_channels=128, num_hidden_convolutions=3

class LineFeatureExtractor(nn.Module):
    
    def __init__(self, cfg:DictConfig) -> None:
        super(LineFeatureExtractor, self).__init__()
        
        self._height = cfg.kernel_height
        self._width = cfg.kernel_width
        
        self._kernels = list(zip(self._height, self._width))

        self._input_channels = cfg.input_channels
        self._hidden_channels = cfg.hidden_channels
        self._output_channels = cfg.output_channels

        self._pool = nn.AdaptiveAvgPool2d(1)

        self._num_middle_conv = cfg.number_of_hidden_convolutions

        self._list_convolutions = []

        self._list_convolutions.append(nn.Conv2d(in_channels=3, out_channels=self._hidden_channels, kernel_size=self._kernels[0], device=device))#(self._min_heigh, self._max_width), device=device))

        for i in range(self._num_middle_conv):
            self._list_convolutions.append(nn.Conv2d(in_channels=self._hidden_channels, out_channels=self._hidden_channels, kernel_size=self._kernels[i+1], device=device))

        self._list_convolutions.append(nn.Conv2d(in_channels=self._hidden_channels, out_channels=self._output_channels, kernel_size=self._kernels[-1], device=device))

        
    def forward(self, x):
        for conv in self._list_convolutions:
            x = conv(x)
            x = torch.relu(x)
            
        x = self._pool(x).view(x.shape[0], -1)

        return x


    
    

        
        


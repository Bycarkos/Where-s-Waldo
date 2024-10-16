
import torch
import torch.nn as nn
from torchvision import transforms


from  torchtyping import TensorType
from jaxtyping import Float, jaxtyped
from math import ceil

import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st 


from typing import *


import pdb


class ProportionalScaling:
    def __init__(
        self,
        shape: Tuple[int, int],
    ) -> None:
        self.height, self.width = shape

    def extract_background_color(self, img):

        channel_ordering = img.view(3, -1)
        

        sorted_pixels, _ = torch.sort(channel_ordering, dim=1, descending=True)

        median_percentile = sorted_pixels[:, :int(sorted_pixels.shape[1] * 0.25)]
        mode_per_channels = torch.mode(median_percentile, dim=1).values

        std_percentile = torch.std(median_percentile, dim=1)
        
        return mode_per_channels, std_percentile



    def __call__(
        self,
        img: Float[torch.Tensor, "channel height width"],
    ) -> Float[torch.Tensor, "channel nw_height nw_width"]:
        


        img = transforms.v2.functional.to_image(img)
        img = transforms.v2.functional.to_dtype(img, torch.float32, scale=True)
        img_channels, img_height, img_width = img.shape
        # print("Original image size: ", img_channels, img_height, img_width)
        mean_color, std_color = self.extract_background_color(img=img)

        factor = min(self.width / img_width, self.height / img_height)
        # logging.debug(f"Resize factor: {factor}")
        new_height, new_width = int(img_height * factor + 0.999), int(
            img_width * factor + 0.999
        )

        # logging.debug(f"New image size: {new_width} {new_height}")

        img = transforms.functional.resize(img, [new_height, new_width])

        padding_up = ceil((self.height - new_height) / 2)
        padding_left = ceil((self.width - new_width) / 2)
        padded = torch.zeros(img_channels, self.height, self.width)#.torch.init.normal_(mean=mean_color, std=std_color)
        for i in range(padded.shape[0]):
            padded[i, :, :] = padded[i, :, :].normal_(mean = mean_color[i], std = std_color[i])

        #print(padded)
        #exit()
        # logging.debug(f"Padding: {padding_left, padding_up}")
        # logging.debug(f"Padded size: {padded.shape}")
        # logging.debug(
        # f"Final coords: {padding_left + new_width, padding_up + new_height}"
        # )

        padded[
            :,
            padding_up : padding_up + new_height,
            padding_left : padding_left + new_width,
        ] = img
        
        return padded




class ThresholdTransform(object):
  def __init__(self, thr_255_chnl):
    self.thr = thr_255_chnl/255   # input threshold for [0..255] gray level, convert to [0..1]

  def __call__(self, x):
    for idx, i in enumerate(self.thr):
        x[idx, :, :] = x[idx, :, :] > i
    
    return (x.mean(0) < .9).to(x.dtype)  # do not change the data type
  

def binarize_background(img: TensorType["C", "H", "W"]):
    channel_ordering = img.view(3, -1)
    sorted_pixels, _ = torch.sort(channel_ordering, dim=1, descending=True)
    median_percentile = sorted_pixels[:, :int(sorted_pixels.shape[1] * 0.25)]
    mode_per_channels = torch.mode(median_percentile, dim=1).values
    
    Th = ThresholdTransform(mode_per_channels)

    mask = Th(img)

    return mask


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
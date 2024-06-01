import torch
import torchvision

from torch.utils.data import Dataset, DataLoader

## Typing packages
from torchtyping import TensorType, patch_typeguard
from typing import *
from pathlib import Path

## folder management packages
import os
import glob


### ocr and info extraction
from PIL import Image
from surya.detection import batch_text_detection
from surya.model.detection.segformer import load_model, load_processor


class Volume():

    def __init__(self, year: int, path: Path, name:str) -> None:
        self._name = name
        self._year = year
        self._path = path

        self._images = []
        self._images_transcriptions = {}



    def number_individuals(self, page_idx):
        """
            page_idx: In this case is the image we want to retrieve from the volume
        """

        page_name = self._images[page_idx]

        return len(self._images_transcriptions[page_name])
    

    def add_page(self, page):
        self._images.append(page)
        
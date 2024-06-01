


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


    

class DocumentLoader(Dataset):

    def __init__(self, root_path: Path) -> None:
        super(DocumentLoader, self).__init__()

        self._volumes = {}
        self._volumes_year = {}

        self._dataset_root_path = root_path

## Custom packages
from data.volumes import Volume, Page, Line, sort_key
import data.dataset_utils as dutils

## geometrical data packages
import torch
import torchvision.transforms.functional as transforms
from torch_geometric.data import HeteroData
from torch.utils.data import Dataset, DataLoader, IterableDataset

## Typing packages
from torchtyping import TensorType, patch_typeguard
from typing import *
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

## commmon packages
import pandas as pd
import os
import glob
from PIL import Image
import itertools
import re
import pdb
from omegaconf import DictConfig, OmegaConf







class ImageDataset(Dataset):

    def __init__(self, Volumes: List[Volume],
                        cfg: DictConfig,
                        mode:str="train"):
        
        super(ImageDataset, self).__init__()

        self._cut_image = cfg.cut_image
        self._patch_size = cfg.patch_size
        
        self._volumes = Volumes

        self._total_attributes_nodes = 0
        self._total_individual_nodes = 0
        self._total_entities = set([])



        for volume in self._volumes:
            self._total_attributes_nodes += torch.tensor(volume.count_attributes())
            self._total_individual_nodes += torch.tensor(volume.count_individuals())
            self._total_entities = self._total_entities.union(set(volume._entities))


        ## merge the grountruth of the different volumes
        self._total_gt = pd.concat([page.gt() for volume in self._volumes for page in volume._pages], axis=0)

        print("Image Lines Recovered with no permutation inconsitences: ", self._total_individual_nodes.item())
        print("Total Lines : ", sum(len(volume.gt()) for volume in self._volumes))
        print("Total proportion : ", (self._total_individual_nodes/sum(len(volume.gt()) for volume in self._volumes)).item(), "%")


        self.extract_information()
        self.standarize_image_shape()

    def standarize_image_shape(self):
        
        max_height = max(self.line_heights)
        min_height = min(self.line_heights)
        if self._cut_image is True: 
            min_width = min(self.line_widths)//2

        else:   
            min_width = min(self.line_widths)
            
        ## reshaping the image
        self.general_width, self.general_height = dutils.extract_optimal_shape(min_width=min_width, max_height=max_height, patch_size=self._patch_size)

        

    def extract_information(self):

        self._page_paths, self._line_paths, self._ocrs = [], [], []
        
        self.line_heights, self.line_widths = [],[]
        self._total_ocrs = []
        self._population_per_volume = []
        total_population = 0
        for iidx, volume in enumerate(self._volumes):
            self._page_paths.extend(volume._pages)
            population_actual_year = 0
            for idx, page in enumerate(volume._pages):
                idx_page = [idx] * len(page._individuals)
                self._line_paths.extend(zip(page._individuals, idx_page))
                population_actual_year += len(page._individuals)
                for line in page._individuals:
                    self._ocrs.append(tuple(line._ocr))
                    self._total_ocrs.extend(line._ocr)
                    w, h = line._bbox[2:]
                    self.line_heights.append(h)
                    self.line_widths.append(w)
        
        
            
            population_belong_volue = np.arange(total_population, total_population + population_actual_year)
            self._population_per_volume.append(population_belong_volue)
        
            total_population += population_actual_year


        self._map_ocr = {ocr: idx for idx, ocr in enumerate(set(self._total_ocrs))}

    def __len__(self):
        return len(self._line_paths)

    def __getitem__(self, idx):
                    
        line, image_page_idx = self._line_paths[idx]

        # extract the pages
        px, py, pw, ph = self._page_paths[image_page_idx]._bbox
        page = self._page_paths[image_page_idx]
        page_cutted = page.image()[py:py+ph, px:px+pw, :]   
        page_cutted = (page_cutted/255).astype(np.float32)    

        ## Extract the lines
        x, y, w, h = line._bbox
        #image_line = page_cutted[y:y+h, x:x+w, :]
        image_line = line.image()
        image_line = (image_line/255).astype(np.float32)
        image_line = torch.from_numpy(image_line)


        image_line = image_line.permute(2, 0, 1)
        _, h, w = image_line.shape

        if w < self.general_width:
            image_line = transforms.resize(img=image_line, size=(self.general_height, self.general_width))
        else:
            image_line = image_line[:, :, :self.general_width]
            image_line = transforms.resize(img=image_line, size=(self.general_height , self.general_width))    

        image_line = transforms.normalize(image_line, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))   

        page_cutted = torch.from_numpy(page_cutted).permute(2, 0, 1)     
        page_cutted = transforms.normalize(page_cutted, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ## Extract the ocr
        ocr = torch.tensor([self._map_ocr[i] for i in self._ocrs[idx]])


        return image_line, ocr, idx


    def collate_fn(self, batch:list):
        """
        Collate function for creating batches.

        Args:
            batch (list): List of samples.

        Returns:
            dict: Dictionary containing batched clips, labels, and paths.
        """

        unbatched_lines, unbatched_ocrs, idx = zip(*batch)

        batched_lines = torch.cat([d.unsqueeze(0) for d in unbatched_lines], dim=0)
        batched_ocr = torch.cat([d.unsqueeze(0) for d in unbatched_ocrs], dim=0)
        batched_population = torch.tensor([d for d in idx])


        return dict(
            image_lines = batched_lines,
            ocrs = batched_ocr,
            population= batched_population
        )


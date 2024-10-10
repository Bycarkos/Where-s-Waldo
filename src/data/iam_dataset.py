## Custom packages


try:
    from data.volumes import Volume, Page, Line, sort_key
    import data.dataset_utils as dutils

except:
    from volumes import Volume, Page, Line, sort_key
    import dataset_utils as dutils


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
import os
import glob
from PIL import Image
import pdb
import tqdm

from torchvision.transforms import v2


from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf




class IAMDataset(Dataset):

    def __init__(self, path: List[Path]) -> None:
                
        super(IAMDataset, self).__init__()

        self.line_heights, self.line_widths = [],[]
        self._line_paths = []
        self._total_ocrs = []
        self._ocrs = []
        sentences_folder = os.path.join(path, "sentences")
        sentences_folders = sorted(os.listdir(sentences_folder))
        print("DOWNLOADING IAM LINES")


        maximun_ocr = 20
        
        with open(os.path.join(path, "ascii", "sentences.txt"), "r") as file:
            transcriptions = file.readlines()[25:]
        
        mapping_transcriptions = {}

        for sentence_transcription in transcriptions:
            tmp = sentence_transcription.split(" ")
            key = tmp[0]

            value = tmp[-1].strip().split("|")

            mapping_transcriptions[key] = value


        for subpart_idx in tqdm.tqdm(range(len(sentences_folders)), bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}'):
            subpart = sentences_folders[subpart_idx]

            id_form = os.path.join(sentences_folder, subpart)
            lines_folders = sorted(glob.glob(id_form+"/**/"))

            for specific_folder_line in lines_folders:
                for _, image_lines in enumerate(sorted(glob.glob(specific_folder_line+ "/*.png"))):
                    
                    mapping_name = os.path.basename(image_lines).split(".")[0]
                    form_page = os.path.dirname(image_lines).split("/")[-1]
                    ocr = mapping_transcriptions[mapping_name]
                    ## necessari  padding
                    ocr += ["PAD"] * (maximun_ocr - len(ocr))
                    self._ocrs.append(tuple(ocr))
                    self._total_ocrs.extend(ocr)

                    self._line_paths.append((Line(_path=image_lines, _bbox=None, _relative_position=None, _ocr=ocr), form_page))
                    w, h = (self._line_paths[-1][0].shape()[0:2])
                    self.line_heights.append(h)
                    self.line_widths.append(w)
        
        self._map_ocr = {ocr: idx for idx, ocr in enumerate(set(self._total_ocrs))}
    
        #self.define_transforms()

    def define_transforms(self, new_shape):

        self._transforms = v2.Compose([

            v2.ToImage(),  
            v2.ToDtype(torch.uint8, scale=True),          
            dutils.ProportionalScaling(new_shape), ## Inner scale
            v2.RandomGrayscale(p=0.4),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
       

        self._non_augmentation_transforms = v2.Compose([
                v2.ToImage(),  
                v2.ToDtype(torch.uint8, scale=True),          
                dutils.ProportionalScaling(new_shape), ## Inner scale
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        

    def __len__(self):
        return len(self._line_paths)

    def __getitem__(self, idx):
                    
        line, image_page_idx = self._line_paths[idx]

        image_line = line.image()
        augmented_image_line = self._transforms(image_line)
        non_augmented_image_line = self._non_augmentation_transforms(image_line)
        copy = self._non_augmentation_transforms(image_line)

        ## Extract the ocr
        ocr = torch.tensor([self._map_ocr[i] for i in self._ocrs[idx]])
        mask = dutils.binarize_background(copy)


        return augmented_image_line, non_augmented_image_line, ocr, idx, mask.repeat(3, 1, 1)


    def collate_fn(self, batch:list):
        """
        Collate function for creating batches.

        Args:
            batch (list): List of samples.

        Returns:
            dict: Dictionary containing batched clips, labels, and paths.
        """

        unbatched_lines, unbatched_non_augmented_lines, unbatched_ocrs, idx, unbatched_mask = zip(*batch)

        batched_lines = torch.cat([d.unsqueeze(0) for d in unbatched_lines], dim=0)
        batched_non_augmented_lines = torch.cat([d.unsqueeze(0) for d in unbatched_non_augmented_lines], dim=0)
        batched_ocr = torch.cat([d.unsqueeze(0) for d in unbatched_ocrs], dim=0)
        batched_population = torch.tensor([d for d in idx])
        bastched_masks = torch.cat([d.unsqueeze(0) for d in unbatched_mask], dim=0)

        return dict(
            image_lines = batched_lines,
            non_augmented_image_lines = batched_non_augmented_lines,
            ocrs = batched_ocr,
            population= batched_population,
            masks = bastched_masks
        )

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    a = IAMDataset(path="/home/cboned/data/HTR/IAM")
    
    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the four lists with different colors
    plt.hist(a.line_widths, color='blue', label='Line Widths 1')
    plt.hist(a.line_heights, color='green', label='Line Heights 1')

    # Add legend
    plt.legend()

    # Add titles and labels
    plt.title('Line Widths and Heights')
    plt.xlabel('Index')
    plt.ylabel('Value')

    # Show the plot
    plt.savefig("dummy_iam.png")


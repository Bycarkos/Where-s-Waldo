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




class WashingtonDataset(Dataset):

    def __init__(self, path: List[Path]) -> None:
                
        super(WashingtonDataset, self).__init__()

        self.line_heights, self.line_widths = [],[]
        self._line_paths = []
        self._total_ocrs = []
        self._ocrs = []
        maximun_ocr = 20

        sentences_folder = os.path.join(path, "data", "line_images_normalized")
        
        sentences_images = sorted(glob.glob(sentences_folder + "/*"))

        sentences_groundtruth = os.path.join(path, "ground_truth", "transcription.txt")

        with open(sentences_groundtruth, "r") as file:
            lines = file.readlines()
        
        print("DOWNLOADING WASHINGTON LINES")
        for idx in tqdm.tqdm(range(len(sentences_images)), bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}'):
            image_path = sentences_images[idx]


            sentence_gt = lines[idx]    
            transcriptions = []

            form_page, notation = sentence_gt.strip().split(' ', 1)
            words_splitted = notation.split("|")

            for word in words_splitted:
                decoded_word = "".join([self.__replace_symbols(token) for token in word.split("-")])
                transcriptions.append(decoded_word)



            ocr = transcriptions +  (["PAD"] * (maximun_ocr - len(transcriptions)))
            self._ocrs.append(tuple(ocr))
            self._total_ocrs.extend(ocr)
            self._line_paths.append((Line(_path=image_path, _bbox=None, _relative_position=None, _ocr=ocr), form_page))
            w, h = (self._line_paths[-1][0].shape()[0:2])
            self.line_heights.append(h)
            self.line_widths.append(w)
    
        self._map_ocr = {ocr: idx for idx, ocr in enumerate(set(self._total_ocrs))}
    
        #self.define_transforms()

    
    
    @staticmethod
    def __replace_symbols(token):
        
        symbol_map = {
        's_cm': ',',
        's_pt': '.',
        "s_qo": ":",
        "s_mi": "_"

        }

        if token in symbol_map:
            # Check if the token is in the custom symbol map first (e.g., s_cm or s_pt)
            return symbol_map[token]
        elif token.startswith('s_'):
            # If token is like s_2, extract the number (it's the last part of the token)
            return token.split("s_")[-1]
        return token  # Return unchanged if it's not a special token

    @staticmethod
    def extract_line_transcriptions(filepath):

        lines_ocrs = []
        with open(filepath, "r") as file:
            lines = file.readlines()

            for line in lines:
                transcriptions = []
                words_splitted = line.strip().split("|")
                for word in words_splitted:
                    decoded_word = "".join([self.__replace_symbols(token) for token in word.split("-")])
                    transcriptions.append(decoded_word)
                
                lines_ocrs.append(transcriptions)

        return lines_ocrs

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

    a = WashingtonDataset(path="/data/users/cboned/data/HTR/Washington")
    exit()
    
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


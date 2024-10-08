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




class EsposallesDataset(Dataset):

    def __init__(self, path: List[Path]) -> None:

        super(EsposallesDataset, self).__init__()

        self.line_heights, self.line_widths = [],[]
        self._line_paths = []
        self._total_ocrs = []
        self._ocrs = []
        subtraining_parts = sorted(os.listdir(path))

        ## Hardcoded
        maximun_ocr = 20

        print("DOWNLOADING ESPOSALLES LINES")
        for subpart_idx in tqdm.tqdm(range(len(subtraining_parts)),  bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}'):
            subpart = subtraining_parts[subpart_idx]
            id_records_path = os.path.join(path, subpart)
            page_records_folders_path = sorted(glob.glob(id_records_path+"/**/"))
            for idPageRecord in page_records_folders_path:
                lines_path = os.path.join(idPageRecord, "lines")
                basename = (os.path.basename(os.path.split(idPageRecord)[0]))
                txt_transcriptions_path = os.path.join(lines_path, basename+"_transcription.txt")
                with open(txt_transcriptions_path, "r") as file:
                    transcriptions = file.readlines()

                page_id = basename.split("_")[0].split("Page")[-1]

                for idx, image_lines in enumerate(sorted(glob.glob(lines_path+ "/*.png"))):
                    
                    ocr = (transcriptions[idx].split(":")[1].split())
                   
                    ## necessari  padding
                    ocr += ["PAD"] * (maximun_ocr - len(ocr))
                    self._ocrs.append(tuple(ocr))
                    self._total_ocrs.extend(ocr)

                    self._line_paths.append((Line(_path=image_lines, _bbox=None, _relative_position=None, _ocr=ocr), int(page_id)))
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

    overrides = [
        "data.dataset.path=../data/CED/SFLL",
        "models.edge_visual_encoder.add_attention=True",
        "models.add_language=False",    
    ]


    with initialize(version_base="1.3.2", config_path="./configs"):
        CFG = compose(config_name="eval", overrides=["data.dataset.path=../data/CED/SFLL"], return_hydra_config=True)
    
    a = EsposallesDataset(path="/home/cboned/data/HTR/Esposalles")
    
    visu.plot(a[0][0])


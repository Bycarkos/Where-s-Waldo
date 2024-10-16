
try:
    from data.volumes import Volume, Page, Line, sort_key
    import data.dataset_utils as dutils

except:
    from volumes import Volume, Page, Line, sort_key
    import dataset_utils as dutils


## geometrical data packages
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

## Typing packages
from typing import *
from pathlib import Path

import numpy as np

## commmon packages
import pandas as pd
import os
import glob
from PIL import Image
import pdb
import json



class CEDDataset(Dataset):

    def __init__(self, path: Path, volumes_years: List[int], attributes:List[str]):
        
        super(CEDDataset, self).__init__()

        self._years_to_use: List[int] = volumes_years
        self._path:Path = path
        self._attributes = attributes

        self._total_attributes_nodes = 0
        self._total_individual_nodes = 0
        self._total_entities = set([])

        self.load_volumes()

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

    def load_volumes(self) -> None:
        
        """
        Load and process volume data from specified paths.

        This function loads volume data from a predefined path, processes the data, and extracts relevant
        information, including ground truth data and visual information for each page in the volumes.

        The function performs the following steps:
        1. Define entities and paths.
        2. Load volume data for specified years.
        3. For each volume, download and process pages.
        4. Extract and process ground truth information for each page.
        5. Create `Volume` objects containing the processed data for each volume.

        This function expects Hydra configuration arguments to specify data paths and other settings.

        Entities:
        
        Data Loading:
            - Load volume data from the specified years.
            - For each volume, iterate through pages and process ground truth data.
            - Extract information such as lines and bounding boxes.
            - Create `Page` objects with the extracted data.

        Returns:
            list: A list of `Volume` objects containing processed page data.

        Example:
            volumes = load_volumes()
            for volume in volumes:
                print(volume)
        """
        
        #~ args should be and hydra config about data
        

        data_path = Path(self._path) #Path("data/CED/SFLL")
        
        
        ## * data loading
        
        data_volumes = [Path(data_path / str(year)) for year in (self._years_to_use)]
        self._volumes = []
        
        for auxiliar_volume in data_volumes:
            print("STARTING DOWNLOADING VOLUMES: VOLUME-", auxiliar_volume)
            pages_path = sorted([f.path for f in os.scandir(auxiliar_volume) if f.is_dir()])
            
            #? Extract groundTruth
            with open(Path(auxiliar_volume, "graph_gt_corroborator.json")) as file:
                graph_gt = json.load(file)
                        
            pages = []
            for idx, page_folder in enumerate(pages_path):
                gt_page = pd.read_csv(page_folder + "/gt_alignement.csv")
                
                ## ?extract the information of the lines first
                page_lines = []
                #page = Image.open(page_folder+".jpg")
                n_families = graph_gt[os.path.basename(page_folder)+".jpg"]["families"]
                
                with open(page_folder + "/info.json", "rb") as file:
                    load_file = json.load(file) 
                    
                    if load_file.get("rows_bbox", None) is None: bboxes = None
                    else: bboxes = load_file["rows_bbox"]


                    if load_file.get("page_bbox", None) is None:
                        page_bboxes = None
                    else:
                        page_bboxes = load_file["page_bbox"] 
                
                #? condition to remove pages with inconsitencis
                if len(glob.glob(os.path.join(page_folder, "row_*"))) != graph_gt[os.path.basename(page_folder)+".jpg"]["individus"]:
                    print(os.path.basename(page_folder)+".jpg")
                    continue
                
                else:
                    percentil_85 = int(np.percentile(np.array(bboxes)[:,-1], 90))

                    lines_page = (glob.glob(os.path.join(page_folder, "row_*")))
                    sorted_lines = sorted(lines_page, key=sort_key)
                    
                    #? Extract groundTruth

                    
                    for idx_line, path_line in enumerate(sorted_lines):
                        
                        
                        ocr = gt_page.loc[idx_line, self._attributes].values
                        bbox_line = bboxes[idx_line]
                        bbox_line[-1] = percentil_85
                        page_lines.append(Line(Path(path_line), bbox_line, bbox_line, ocr))
                
                pages.append(Page(Path(page_folder+".jpg"), page_bboxes, page_lines, n_families))
                
            self._volumes.append(Volume(auxiliar_volume, pages, auxiliar_volume, self._attributes))



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
                    w, h = (self._line_paths[-1][0].shape()[0:2])
                    self.line_heights.append(h)
                    self.line_widths.append(w)  
            
            population_belong_volue = np.arange(total_population, total_population + population_actual_year)
            self._population_per_volume.append(population_belong_volue)
        
            total_population += population_actual_year


        self._map_ocr = {ocr: idx for idx, ocr in enumerate(set(self._total_ocrs))}


    def define_page_transforms(self, new_shape):
        self._page_augmentation = v2.Compose([
                v2.ToImage(),  
                v2.ToDtype(torch.uint8, scale=True),          
                dutils.ProportionalScaling(new_shape), ## Inner scale
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

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
    
    def _getpage(self, idx):

        _, image_page_idx = self._line_paths[idx]
        page = self._page_paths[image_page_idx]
        image_page = page.image()
        augmented_page_image = self._transforms(image_page)

        return augmented_page_image

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


    a = CEDDataset(path="/home/cboned/data/HTR/CED/SFLL", volumes_years=[1889], attributes=["nom"])
    a.define_transforms(new_shape=(50, 1500))
    print(a[0][0].shape)


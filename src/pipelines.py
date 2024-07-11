## Dataset Things
from data.volumes import Volume, Page, Line
from data.dataset import Graphset
from data.collator import FamilyCollator

import data.volumes as dv


## Model Things
from models import gnn_encoders as gnn
from models import visual_encoders as cnn


## Common packages
import torch.nn as nn
import torchvision.transforms as T

## Typing Packages
from typing import *


## Configuration Package
from omegaconf import DictConfig, OmegaConf


import tqdm

## Common packages
import os
import json
import glob
import pandas as pd # type: ignore
from pathlib import Path
from PIL import Image



"""
Red (!)
Blue (?)
Green (*)
Yellow (^)
Pink (&)
Purple (~)
Mustard (todo)
Grey (//)
Commented out Code can also be Styled to make it Clear the Code shouldn't be There.
Any Other Comment Styles you'd like can be Specified in the Settings.
"""



def load_volumes(cfg: DictConfig):
    
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

    Returns:
        volums (list): A list of `Volume` objects, each containing pages and associated metadata.

    Hydra Configuration Args:
        This function expects Hydra configuration arguments to specify data paths and other settings.

        Entities:
        Data Path:
        Volume Years:

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
    

    entities = cfg.entities #["nom", "cognom_1", "cognom_2", "parentesc", "ocupacio"]

    data_path = Path(cfg.path) #Path("data/CED/SFLL")
    
    volume_years = cfg.volume_years #[1889, 1906]
    
    ## * data loading
    
    data_volumes = [Path(data_path / str(year)) for year in (volume_years)]
    volums = []
    
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
            page = Image.open(page_folder+".jpg")
            n_families = graph_gt[os.path.basename(page_folder)+".jpg"]["families"]
            
            #? condition to remove pages with inconsitencis
            if len(glob.glob(os.path.join(page_folder, "row_*"))) != graph_gt[os.path.basename(page_folder)+".jpg"]["individus"]:
                continue
            
            else:
                lines_page = (glob.glob(os.path.join(page_folder, "row_*")))
                sorted_lines = sorted_file_names = sorted(lines_page, key=dv.sort_key)
                
                #? Extract groundTruth
                with open(page_folder + "/info.json", "rb") as file:
                    bboxes = json.load(file)["rows_bbox"]
                    
                for idx_line, path_line in enumerate(sorted_lines):
                    ocr = gt_page.loc[idx_line, entities].values
                    bbox_line = bboxes[idx_line]
                    page_lines.append(Line(Path(path_line), bbox_line, bbox_line, ocr))
            
            pages.append(Page(Path(page_folder+".jpg"), [0, 0, *page.size ], page_lines, n_families))
            
        volums.append(Volume(auxiliar_volume, pages, entities))

    return volums


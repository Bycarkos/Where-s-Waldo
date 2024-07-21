## Dataset Things
from data.volumes import Volume, Page, Line
from data.dataset import Graphset
from data.collator import FamilyCollator

import data.volumes as dv


## Model Things
from models import gnn_encoders as gnn
from models import visual_encoders as cnn
from models.graph_construction_model import MMGCM



## Common packages
import torch.nn as nn
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as transforms

from torch_geometric.data import HeteroData

## Typing Packages
from typing import *
from torchtyping import TensorType

## Configuration Package
from omegaconf import DictConfig, OmegaConf


import tqdm

## Common packages
import os
import json
import glob
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import copy
import pickle
import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
import wandb
import cv2 as cv



import umap
import umap.plot

device = "cuda" if torch.cuda.is_available() else "cpu"
import pprint
from beeprint import pp as bprint



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





def batch_step(loader:Type[FamilyCollator], 
               graph_structure:Type[HeteroData], 
               model: Type[nn.Module],
               optimizer: Type[torch.optim.Adam],
               criterion,
               image_reshape: Tuple[int, int], 
               entities: Tuple[str, ...],
               epoch: int, 
               mode:str="train"):

        """
        Perform a single training or evaluation step on a batch of data.

        Parameters:
        -----------
        loader : Type[FamilyCollator]
            Data loader that provides batches of edge indices, negative edge indices, and populations.
        graph_structure : Type[HeteroData]
            Graph structure containing node features and edge indices.
        model : Type[nn.Module]
            The model to be trained or evaluated.
        optimizer : Type[torch.optim.Adam]
            Optimizer for updating the model's parameters.
        criterion : function
            Loss function used to compute the task loss.
        image_reshape : Tuple[int, int]
            Tuple containing the desired height and width for reshaping images.
        entities : Tuple[str, ...]
            Tuple containing the names of the entities for which losses are computed.
        epoch : int
            The current epoch number.
        mode : str, optional (default="train")
            Mode of operation, either "train" or "eval".

        Returns:
        --------
        float
            The total loss for the current epoch.
        """




        if mode == "train":
            model.train()
            
        else:
            model.eval()

        height, width = image_reshape
        epoch_train_loss = 0
        epoch_entities_losses = {key+"_"+mode+"_loss":0 for key in entities}

        for idx, (edge_index_dict, negative_edge_index_dict, population) in tqdm.tqdm(enumerate(loader), ascii=True):
            optimizer.zero_grad()
            ### All the time work with the same x_dict
            x_dict = graph_structure.x_dict    
            images_to_keep = []

            for individual_index in population:
                image = graph_structure["image_lines"][individual_index]._path
                image = cv.imread(image)
                #plt.imsave("./plots/original.jpg", image[:,:,::-1])
                
                image = transforms.to_tensor(image)
                _, h, w = image.shape

                if w < width:
                    image = transforms.resize(img=image, size=(height, width))
                    
                else:
                    image = image[:, :, :width]
                    image = transforms.resize(img=image, size=(height, width))
                    
                
                image = image[:, :, :width//2]
                plotting_image = torch.permute(image, (1,2,0))
                #plt.imsave("./plots/test.jpg", plotting_image.numpy()[:,:,::-1])

                images_to_keep.append(image)

            ## Send information to the GPU
            batch = torch.from_numpy(np.array(images_to_keep)).to(device=device)
            population = population.to(device=device)
            x_dict = {key: value.to(device) for key, value in x_dict.items()}
            edge_index_dict = {key: value.to(device) for key, value in edge_index_dict.items()}
            
            if mode == "train":
                ## Extract visual information
                individual_features = model.encode_visual_information(x=batch)
                x_dict["individuals"][population] = individual_features  # update the information of the individuals 

                if model._apply_edges is not False:
                    edge_features = model.encode_edge_positional_information(x=batch)    
                else:
                    edge_features = None
                             
                x_dict = model.encode_attribute_information(x_dict=x_dict, edge_index_dict=edge_index_dict, edge_attributes=edge_features, population=population) #message_passing
                
            else:
                with torch.no_grad():
                    individual_features = model.encode_visual_information(x=batch)
                    x_dict["individuals"][population] = individual_features  # update the information of the individuals 
                    if model._apply_edges is not False:
                        edge_features = model.encode_edge_positional_information(x=batch)    
                    else:
                        edge_features = None
                    
                    x_dict = model.encode_attribute_information(x_dict=x_dict, edge_index_dict=edge_index_dict, edge_attributes=edge_features, population=population) #message_passing
                    
            ### *Task loss
            loss = 0
            batch_entites_loss = copy.copy(epoch_entities_losses)
            for attribute in entities:

                edge_similar_name = (attribute, "similar", attribute)
                edge_index_similar = edge_index_dict[edge_similar_name].to(device=device)
                
                
                positive_labels = torch.ones(edge_index_similar.shape[1])
                
                negative_edge_index_similar = negative_edge_index_dict[edge_similar_name].to(device=device)
                negative_labels = torch.zeros(negative_edge_index_similar.shape[1])
                
                gt = torch.cat((positive_labels, negative_labels), dim=0).to(device=device)
                edge_index = torch.cat((edge_index_similar, negative_edge_index_similar), dim=1)
                x1 = x_dict[attribute][edge_index[0,:]]
                x2 = x_dict[attribute][edge_index[1, :]]
                actual_loss = criterion(x1, x2, gt)
                loss += actual_loss
                batch_entites_loss[attribute+"_"+mode+"_loss"] += actual_loss

            
            
            if mode == "train":
                loss.backward()
                optimizer.step()
            
            
        
        epoch_train_loss += loss
        epoch_entities_losses.update(batch_entites_loss)
        epoch_entities_losses[mode+"_loss"] = epoch_train_loss

        print(f"Epoch {epoch}: Loss: {epoch_train_loss}")

        wandb.log(epoch_entities_losses)

        return epoch_train_loss
    
    
    
@torch.no_grad
def evaluate_attribute_metric_space(collator:Type[FamilyCollator],
                                    graph_structure:Type[HeteroData], 
                                    model: Type[nn.Module],
                                    image_reshape: Tuple[int, int], 
                                    entities: Tuple[str, ...]):

    height, width = image_reshape
    model.eval()
    final_dict = {"train": {}, "test": {}}
    test_dict = {}
    population_dict = {"train": [], "test": []}
    for mode in ["train", "test"]:
        collator._change_mode(mode=mode)
        for idx, (edge_index_dict, _, population) in tqdm.tqdm(enumerate(collator), ascii=True):
            
            
            x_dict = graph_structure.x_dict    
            images_to_keep = []

            for individual_index in population:
                population_dict[mode].append(individual_index.item())
                image = graph_structure["image_lines"][individual_index]._path
                image = cv.imread(image)
                #plt.imsave("./plots/original.jpg", image[:,:,::-1])
                
                image = transforms.to_tensor(image)
                _, h, w = image.shape

                if w < width:
                    image = transforms.resize(img=image, size=(height, width))
                    
                else:
                    image = image[:, :, :width]
                    image = transforms.resize(img=image, size=(height, width))
                    
                
                image = image[:, :, :width//2]
                plotting_image = torch.permute(image, (1,2,0))
                #plt.imsave("./plots/test.jpg", plotting_image.numpy()[:,:,::-1])

                images_to_keep.append(image)
    
        
        ## Send information to the GPU
            batch = torch.from_numpy(np.array(images_to_keep)).to(device=device)
            population = population.to(device=device)
            x_dict = {key: value.to(device) for key, value in x_dict.items()}
            edge_index_dict = {key: value.to(device) for key, value in edge_index_dict.items()}
            
            individual_features = model.encode_visual_information(x=batch)
            x_dict["individuals"][population] = individual_features  # update the information of the individuals 
            
            if model._apply_edges is not False:
                edge_features = model.encode_edge_positional_information(x=batch)    
            else:
                edge_features = None
            
            x_dict = model.encode_attribute_information(x_dict=x_dict, edge_index_dict=edge_index_dict, edge_attributes=edge_features, population=population) #message_passing
        
            final_dict[mode].update(x_dict)
                
                
                
    print(final_dict["test"].keys())
    exit()
            
    
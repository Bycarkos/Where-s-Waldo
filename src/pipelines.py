## Dataset Things
from data.volumes import Volume, Page, Line
from data.graphset import Graphset

import data.volumes as dv


## Model Things
from models import gnn_encoders as gnn
from models import visual_encoders as cnn
from models.graph_construction_model import MMGCM


import utils
import visualizations as visu
import tasks.record_linkage as rl

## Common packages
import torch.nn as nn
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as transforms
from torch.utils.data import DataLoader

from torch_geometric.data import HeteroData
import torch_geometric.utils as tutils


from pytorch_metric_learning import miners, losses



## Typing Packages
from typing import *
from torchtyping import TensorType

## Configuration Package
from omegaconf import DictConfig, OmegaConf


import tqdm

import fasttext

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
import itertools
import random


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



import umap
import umap.plot

device = "cuda" if torch.cuda.is_available() else "cpu"
import pprint
from beeprint import pp as bprint
import pdb



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
    

    entities = cfg.graph_configuration.attribute_type_of_nodes #["nom", "cognom_1", "cognom_2", "parentesc", "ocupacio"]

    data_path = Path(cfg.dataset.path) #Path("data/CED/SFLL")
    
    volume_years = cfg.dataset.volumes #[1889, 1906]
    
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
            #page = Image.open(page_folder+".jpg")
            n_families = graph_gt[os.path.basename(page_folder)+".jpg"]["families"]
            
            with open(page_folder + "/info.json", "rb") as file:
                load_file = json.load(file) 
                if load_file.get("rows_bbox", None) is None:
                    bboxes = None
                else:
                    bboxes = load_file["rows_bbox"]


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
                sorted_lines = sorted(lines_page, key=dv.sort_key)
                
                #? Extract groundTruth

                
                for idx_line, path_line in enumerate(sorted_lines):
                    
                    
                    ocr = gt_page.loc[idx_line, entities].values
                    bbox_line = bboxes[idx_line]
                    bbox_line[-1] = percentil_85
                    page_lines.append(Line(Path(path_line), bbox_line, bbox_line, ocr))
            
            pages.append(Page(Path(page_folder+".jpg"), page_bboxes, page_lines, n_families))
            
        volums.append(Volume(auxiliar_volume, pages, auxiliar_volume, entities))

    return volums






@torch.no_grad()
def evaluate(loader: Type[DataLoader],
             graph: Type[HeteroData],
             model: Type[nn.Module],
             criterion: Type[nn.Module],
             entities:list,
             language_distillation:bool=False):


    model.eval()

    attribute_embeddings = graph.x_attributes.to(device)
    entity_embeddings = graph.x_entity.to(device)

    core_graph = graph
    if language_distillation:
        x_ocr = torch.from_numpy(np.array(graph.x_language)).to(device)


    ## Attribute embeddings Extraction 
    for idx, dict_images in tqdm.tqdm(enumerate(loader), desc="Extracting the embeddings from the model"):
        images = dict_images["image_lines"].to(device)
        ocr_indexes = dict_images["ocrs"].to(device)
        population = dict_images["population"].to(device)

        attribute_representation, individual_embeddings = model(x=images)#.encode_attribute_information(image_features=image_features, edge_attributes=edge_features) #message_passing
        attribute_embeddings[population] = attribute_representation
        if individual_embeddings is not None:
            entity_embeddings[population, 0] = individual_embeddings
            
            
        loss = 0
        edges_to_keep = [(attribute, "similar", attribute) for attribute in entities]
        subgraph = utils.extract_subgraph(graph=graph, population=population, edges_to_extract=edges_to_keep)
        for idx, attribute in enumerate(entities):
            

            edge_similar_name = (attribute, "similar", attribute)
            similar_population = subgraph[edge_similar_name].flatten().unique().to(device)                
            labels = torch.isin(population, similar_population).type(torch.int32)
            if attribute == "individual":
                embeddings = individual_embeddings
                
            else:
                embeddings = attribute_representation[:, idx,:]
            
                
                if language_distillation:

                    selected_language_embeddings_idx = ocr_indexes[:, idx].type(torch.int32) 
                    language_embeddings = x_ocr[selected_language_embeddings_idx,:]

                    labels = torch.cat((labels, labels), dim=0)
                    embeddings = torch.cat((embeddings, language_embeddings), dim=0)

            loss_attribute = criterion(embeddings, labels)
            loss += (loss_attribute)
        

    print("VALIDATION LOSS: ", loss)
    
    return loss







def batch_step(loader, 
               graph:Type[HeteroData], 
               model: Type[nn.Module],
               optimizer: Type[torch.optim.Adam],
               criterion,
               entities: Tuple[str, ...],
               epoch: int,
               language_distillation:bool=False):

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


        

        model.train()
            
        miner = miners.PairMarginMiner(pos_margin=0.2, neg_margin=0.8)
        


        criterion = losses.TripletMarginLoss(margin=0.2,
                        swap=False,
                        smooth_loss=False,
                        triplets_per_anchor="all")
        
        epoch_train_loss = 0
        epoch_entities_losses = {key+"_train_loss":0 for key in entities}

        if language_distillation: 
            x_ocr = torch.from_numpy(np.array(graph.x_language)).to(device)

        
        for idx, dict_images in tqdm.tqdm(enumerate(loader), ascii=True):
            optimizer.zero_grad()

            images = dict_images["image_lines"].to(device)
            ocr_indexes = dict_images["ocrs"].to(device)
            population = dict_images["population"].to(device)

            attribute_representation, individual_embeddings = model(x=images)#.encode_attribute_information(image_features=image_features, edge_attributes=edge_features) #message_passing
            
                
            ### *Task loss
            loss = 0
            batch_entites_loss = copy.copy(epoch_entities_losses)
            edges_to_keep = [(attribute, "similar", attribute) for attribute in entities]
            subgraph = utils.extract_subgraph(graph=graph, population=population, edges_to_extract=edges_to_keep)
            for idx, attribute in enumerate(entities):
                

                edge_similar_name = (attribute, "similar", attribute)
                similar_population = subgraph[edge_similar_name].flatten().unique().to(device)                
                labels = torch.isin(population, similar_population).type(torch.int32)
                if attribute == "individual":
                    embeddings = individual_embeddings
                    
                else:
                    embeddings = attribute_representation[:, idx,:]

                    
                    if language_distillation:         
                        selected_language_embeddings_idx = ocr_indexes[:, idx].type(torch.int32) 
                        language_embeddings = x_ocr[selected_language_embeddings_idx,:]
                        labels = torch.cat((labels, labels), dim=0)
                        embeddings = torch.cat((embeddings, language_embeddings), dim=0)

                loss_attribute = criterion(embeddings, labels)
                loss += (loss_attribute) # + contrastive_loss_attribute) #distilattion_loss
                
                batch_entites_loss[attribute+"_train_loss"] += loss_attribute

            loss.backward()
            optimizer.step()
            
            
        
        epoch_train_loss += loss
        epoch_entities_losses.update(batch_entites_loss)
        epoch_entities_losses["train_loss"] = epoch_train_loss

        print(f"Epoch {epoch}: Loss: {epoch_train_loss}")

        wandb.log(epoch_entities_losses)
        
        
        return loss
    
    
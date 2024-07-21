## Dataset Things
from data.volumes import Volume, Page, Line
from data.dataset import Graphset
from data.collator import FamilyCollator

import data.volumes as dv

## Model Things
from models import gnn_encoders as gnn
from models import visual_encoders as cnn
from models.graph_construction_model import MMGCM


### Utils
import utils 


## Pipelines
import pipelines as pipes


## Common packages
import torch.nn as nn
import torchvision.transforms.functional as transforms
import torch
from vit_pytorch.na_vit import NaViT

from torch.optim import Adam
from torch_geometric.nn import to_hetero

import numpy as np

## Typing Packages
from typing import *
from torchtyping import  TensorType
from torch_geometric.data import HeteroData


## Configuration Package
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

## Experiment Tracking packages
import wandb
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


import umap
import umap.plot

device = "cuda" if torch.cuda.is_available() else "cpu"
import pprint
from beeprint import pp as bprint



@hydra.main(config_path="./configs", config_name="main", version_base="1.3.2")
def main(cfg: DictConfig):
    
    cfg_model = cfg.models
    cfg_data = cfg.data
    cfg_setup = cfg.setup


    volumes = pipes.load_volumes(cfg=cfg_data.dataset)
    
    #! The utility of this dictionary is to relate the groundtruth with the visual information AD-HOC
    pk = {"Noms_harmo":"nom", "cognom1_harmo":"cognom_1", "cognom2_harmo":"cognom_2", "parentesc_har":"parentesc", "ocupacio":"ocupacio"}
    
    #  ^ Hydra things
    batch_size = cfg_data.collator.batch_size
    shuffle = cfg_data.collator.shuffle
    embedding_size = cfg_setup.config.embedding_size
    patch_size = cfg_setup.config.patch_size
    epochs = cfg_setup.config.epochs
    entities = cfg_data.dataset.entities
    # ^ 
    
    if os.path.exists("./pickles/graphset.pkl"):
        with open("./pickles/graphset.pkl", "rb") as file:
            Graph = pickle.load(file)

    else:
        Graph = Graphset(Volumes=volumes, auxiliar_entities_pk=pk, add_edge_features=True)
        Graph._initialize_image_information()
        Graph._initialize_edges()
            
    Graph._initialize_nodes(embedding_size = embedding_size)

    ##? 

    #? Define the Collator
    ## ! Change the way to initialize the collator (we need to change the mode because we are creating 3 instances)
    collator = FamilyCollator(graph=Graph, batch_size=batch_size, shuffle=shuffle, mode="train")
    collator_test = copy.copy(collator)
    collator_test._change_mode("test")
    
    core_graph = Graph._graph
    #? 
    
    ## ** model things
    ratio_kernel = cfg_model.kernel_ratio

    min_width = core_graph["mean_width"]
    max_heigh = core_graph["min_height"] + core_graph["mean_height"]
    
    width, height = utils.extract_optimal_shape(min_width=min_width, max_height=max_heigh, patch_size=patch_size) 
    width = int(width.item())
    height = int(height.item())
    
    kernel_height = int((height*ratio_kernel))
    kernel_width = 5

    ## ** 
    
    ## ^ Model 
    cfg_model.line_encoder.kernel_height = kernel_height
    cfg_model.line_encoder.kernel_width = kernel_width
    cfg_model.gnn_encoder.attributes = cfg_data.dataset.entities
    cfg_model.edge_encoder.number_of_entities = len(entities)

    if cfg_model.apply_edge_encoder is False:
        model = MMGCM(visual_encoder=cnn.LineFeatureExtractor, gnn_encoder=gnn.FamilyAttributeGnn, cfg=cfg_model)
    
    else:
        model = MMGCM(visual_encoder=cnn.LineFeatureExtractor, gnn_encoder=gnn.FamilyAttributeGnn, edge_encoder=cnn.EdgeAttFeatureExtractor, cfg=cfg_model)

    
    dataset_name = cfg_data.dataset.name
    model_name = f"./checkpoints/{dataset_name}_{cfg.name_checkpoint}.pt"

    model.load_state_dict(torch.load(model_name))
    model.to(device)
    
    print("MODEL LOADED SUCCESFULLY")
    
    
    pipes.evaluate_attribute_metric_space(collator=collator,
                                          graph_structure=core_graph,
                                          model=model,
                                          image_reshape=(height, width),
                                          entities=entities)
    
    
        
if __name__ == "__main__":

    main()
        
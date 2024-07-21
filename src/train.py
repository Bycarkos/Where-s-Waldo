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


"""
                if mode == "validation" or mode == "test":
                    previous_x_dict = graph_structure.x_dict   
                    updated_dict = x_dict 
                    
                    dirname = f"./plots/shifts/{attribute}/{mode}"
                    name_file = f"{attribute}_epoch_{epoch}_{idx}_drift.jpg"
                    fig_path = os.path.join(dirname, name_file)
                    dict_drift[name_file] = population.cpu().numpy()
                    
                    
                    
                    x_previous = previous_x_dict[attribute].detach().cpu().numpy()
                    x_after = updated_dict[attribute].detach().cpu().numpy()
                            
                    evaluate_shift(x_previous=x_previous, x_after=x_after, population=population.cpu().numpy(), fig_name=fig_path)
      
      
            if mode == "validation" or mode == "test":
                with open(f"{pickles_path}/drift_corresp.pkl", "ab") as file:
                    pickle.dump(dict_drift, file)          
"""

    
    
    
@hydra.main(config_path="./configs", config_name="main", version_base="1.3.2")
def main(cfg: DictConfig):
    
    cfg_model = cfg.models
    cfg_data = cfg.data
    cfg_setup = cfg.setup

    if cfg.log_wandb == True:

        if cfg.verbose == True :
            print("Wandb configuration: ")

            bprint(dict(cfg_setup.wandb))

        wandb.login(key="ab18aafa8c70616ba4ef66844fc9444794cae54a", relogin=True)

        wandb.init(
            project= cfg_setup.wandb.project,
            config = dict(cfg_setup.wandb.config),
            notes = cfg_setup.wandb.notes,
            name = cfg_setup.wandb.name,
            group = cfg_setup.wandb.group
        )

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
    
    #? Define the dataset
    if  cfg_data.import_data == True:
        if os.path.exists("./pickles/graphset.pkl"):
            with open("./pickles/graphset.pkl", "rb") as file:
                Graph = pickle.load(file)
    
        else:
            Graph = Graphset(Volumes=volumes, auxiliar_entities_pk=pk, add_edge_features=True)
            Graph._initialize_image_information()
            Graph._initialize_edges()
            
    else:
    
        Graph = Graphset(Volumes=volumes, auxiliar_entities_pk=pk, add_edge_features=True)
        Graph._initialize_image_information()
        Graph._initialize_edges()
        
    if  cfg_data.export_data == True:

        os.makedirs("./pickles", exist_ok=True)
        with open("./pickles/graphset.pkl", "wb") as file:
            pickle.dump(Graph, file)
        
    
    Graph._initialize_nodes(embedding_size = embedding_size)

    ##? 

    #? Define the Collator
    ## ! Change the way to initialize the collator (we need to change the mode because we are creating 3 instances)
    collator = FamilyCollator(graph=Graph, batch_size=batch_size, shuffle=shuffle, mode="train")
    #collator_validation = FamilyCollator(graph=Graph, batch_size=batch_size, shuffle=shuffle, mode="validation")
    #collator_test = FamilyCollator(graph=Graph, batch_size=batch_size, shuffle=shuffle, mode="test")



    core_graph = Graph._graph
    similar_edge_index=core_graph[("nom", "similar", "nom")].edge_index
    image_lines =  np.array(core_graph["image_lines"])
    #? 
    
    ## ** model things
    ratio_kernel = cfg_model.kernel_ratio

    min_width = core_graph["mean_width"]
    max_heigh = core_graph["min_height"] + core_graph["mean_height"]
    
    width, height = utils.extract_optimal_shape(min_width=min_width, max_height=max_heigh, patch_size=patch_size) 
    width = int(width.item())
    height = int(height.item())
    
    kernel_height = int((height*ratio_kernel))
    kernel_width = 5#int(kernel_height * (1-ratio_kernel))

    ## ** 
    
    ## ^ Model 
    cfg_model.line_encoder.kernel_height = kernel_height
    cfg_model.line_encoder.kernel_width = kernel_width
    cfg_model.gnn_encoder.attributes = cfg_data.dataset.entities
    cfg_model.edge_encoder.number_of_entities = len(entities)

    if cfg_model.apply_edge_encoder is False:
        model = MMGCM(visual_encoder=cnn.LineFeatureExtractor, gnn_encoder=gnn.FamilyAttributeGnn, cfg=cfg_model).to(device)
    
    else:
        model = MMGCM(visual_encoder=cnn.LineFeatureExtractor, gnn_encoder=gnn.FamilyAttributeGnn, edge_encoder=cnn.EdgeAttFeatureExtractor, cfg=cfg_model).to(device)

    
    #params = list(Line_Encoder.parameters()) + list(Edge_Positional_Encoder.parameters()) + list(Gnn_Encoder.parameters())
    
    optimizer = hydra.utils.instantiate(cfg_setup.optimizer, params=model.parameters())
    criterion = utils.contrastive_loss
    
    model.train()

    if cfg.verbose == True:
        print("Configuration of the Models: ")
        bprint(dict(cfg_model))

    if cfg.verbose == True:
        print("Inizialization Done without problemas")
        print("Starting the training proces with the following configuration: ")
        bprint(dict(cfg_setup.config)) 
    
    ## ^  

    optimal_loss = 10000

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        
        if epoch == 0:
            collator._change_mode(mode="validation")
            validation_loss = pipes.batch_step(loader=collator, graph_structure=core_graph, 
                                    model=model, 
                                    criterion=criterion, 
                                    optimizer=optimizer,
                                    image_reshape=(height, width), 
                                    entities=entities, 
                                    epoch=epoch, 
                                    mode="validation")
            
            if validation_loss < optimal_loss:
                os.makedirs("./checkpoints", exist_ok=True)
                dataset_name = cfg_data.dataset.name
                model_name = f"./checkpoints/{dataset_name}_{cfg.name_checkpoint}.pt"


                torch.save(model.state_dict(), model_name)

        collator._change_mode(mode="train")
        train_loss  = pipes.batch_step(loader=collator, graph_structure=core_graph, 
                                    model=model, 
                                    criterion=criterion, 
                                    optimizer=optimizer,
                                    image_reshape=(height, width), 
                                    entities=entities, 
                                    epoch=epoch, 
                                    mode="train")

        
        if (epoch +1) % 10 == 0:
            collator._change_mode(mode="validation")
            validation_loss = pipes.batch_step(loader=collator, graph_structure=core_graph, 
                                    model=model, 
                                    criterion=criterion, 
                                    optimizer=optimizer,
                                    image_reshape=(height, width), 
                                    entities=entities, 
                                    epoch=epoch, 
                                    mode="validation")
            
            if validation_loss < optimal_loss:
                os.makedirs("./checkpoints", exist_ok=True)
                dataset_name = cfg_data.dataset.name
                model_name = f"./checkpoints/{dataset_name}_{cfg.name_checkpoint}.pt"

                torch.save(model.state_dict(), model_name)

             

    collator._change_mode(mode="test")
    test_loss = pipes.batch_step(loader=collator, graph_structure=core_graph, 
                                    model=model, 
                                    criterion=criterion,
                                    optimizer=optimizer, 
                                    image_reshape=(height, width), 
                                    entities=entities, 
                                    epoch=epoch, 
                                    mode="test")         
    
    if test_loss < optimal_loss:
        os.makedirs("./checkpoints", exist_ok=True)
        dataset_name = cfg_data.dataset.name
        model_name = f"./checkpoints/{dataset_name}_{cfg.name_checkpoint}.pt"

        torch.save(model.state_dict(), model_name)
    
    wandb.finish()

        
        
if __name__ == "__main__":

    main()
        
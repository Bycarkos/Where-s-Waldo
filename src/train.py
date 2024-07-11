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


def batch_step(loader:Type[FamilyCollator], 
               graph_structure:Type[HeteroData], 
               Line_Encoder: Type[nn.Module], 
               Edge_Positional_Encoder: Type[nn.Module],
               Gnn_Encoder: Type[gnn.HeteroGNN],
               optimizer: Type[torch.optim.Adam],
               criterion,
               image_reshape: Tuple[int, int], 
               entities: Tuple[str, ...],
               epoch: int, 
               mode:str="train"):

        height, width = image_reshape
        epoch_train_loss = 0
        epoch_entities_losses = {key+"_"+mode+"_loss":0 for key in entities}
        with torch.autograd.set_detect_anomaly(True):
            for idx, (edge_index_dict, negative_edge_index_dict, population) in (enumerate(loader)):
                optimizer.zero_grad()
                ### All the time work with the same x_dict
                x_dict = graph_structure.x_dict    
                images_to_keep = []

                for individual_index in population:
                    image = graph_structure["image_lines"][individual_index].image()
                    image = transforms.to_tensor(image)
                    _, h, w = image.shape

                    if w < width:
                        image = transforms.resize(img=image, size=(height, width))
                        
                    else:
                        image = image[:, :, :width]
                        image = transforms.resize(img=image, size=(height, width))
                    
                    images_to_keep.append(image)


                batch = torch.from_numpy(np.array(images_to_keep)).to(device=device)
                population = population.to(device=device)
                x_dict = {key: value.to(device) for key, value in x_dict.items()}
                edge_index_dict = {key: value.to(device) for key, value in edge_index_dict.items()}
                
                ## Extract visual information
                individual_features = Line_Encoder(x=batch)
                edge_features = Edge_Positional_Encoder(x=batch)
                
                # update the information of the individuals 
                x_dict["individuals"][population] = individual_features
                
                features_dict = Gnn_Encoder(x_dict, edge_index_dict, edge_features, population)

                x_dict.update(features_dict)

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
                    loss = loss + criterion(x1, x2, gt)
                    batch_entites_loss[attribute+"_"+mode+"_loss"] += loss

                loss.backward()
                optimizer.step()
            
        epoch_train_loss += loss
        epoch_entities_losses.update(batch_entites_loss)

        epoch_entities_losses[mode+"_loss"] = epoch_train_loss

        print(f"Epoch {epoch}: Loss: {epoch_train_loss}")

        wandb.log(epoch_entities_losses)
        









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
    embedding_size = cfg_setup.config.batch_size
    patch_size = cfg_setup.config.patch_size
    epochs = cfg_setup.config.epochs
    # ^ 
    
    #? Define the dataset
    Graph = Graphset(Volumes=volumes, auxiliar_entities_pk=pk, add_edge_features=True)
    Graph._initialize_nodes(embedding_size = embedding_size)
    Graph._initialize_image_information()
    Graph._initialize_edges()
    ##? 

    #? Define the Collator
    collator = FamilyCollator(graph=Graph, batch_size=batch_size, shuffle=shuffle)
    train_loader = collator.collate_train()
    val_loader = collator.collate_validation()
    test_loader = collator.collate_test()

    core_graph = Graph._graph
    #? 
    
    ## ** model things
    ratio_kernel = cfg_model.kernel_ratio

    min_width = core_graph["mean_width"]
    max_heigh = core_graph["min_height"] + core_graph["mean_height"]
    
    width, height = utils.extract_optimal_shape(min_width=min_width, max_height=max_heigh, patch_size=patch_size) 
    width = int(width.item())
    height = int(height.item())
    
    ratio = width /height
    
    kernel_height = int((height*ratio_kernel))
    kernel_width = int(kernel_height * (1-ratio_kernel))


    line_encoder = cfg_model.line_encoder
    edge_encoder = cfg_model.edge_encoder
    gnn_encoder = cfg_model.gnn_encoder
    ## ** 
    
    ## ^ Model 
    line_encoder.kernel_height = kernel_height
    line_encoder.kernel_width = kernel_width



    Line_Encoder = cnn.LineFeatureExtractor(cfg=line_encoder).to(device)
    
    Edge_Positional_Encoder = cnn.EdgeAttFeatureExtractor(cfg=edge_encoder).to(device=device)
    
    
    gnn_encoder.attributes = list(pk.values())
    Gnn_Encoder = gnn.HeteroGNN(cfg = gnn_encoder).to(device=device)

    params = list(Line_Encoder.parameters()) + list(Edge_Positional_Encoder.parameters()) + list(Gnn_Encoder.parameters())
    
    optimizer = hydra.utils.instantiate(cfg_setup.optimizer, params=params)
    criterion = utils.contrastive_loss
    
    Line_Encoder.train()
    Edge_Positional_Encoder.train()
    Gnn_Encoder.train() 

    if cfg.verbose == True:
        print("Configuration of the Models: ")
        bprint(dict(cfg_model))



    if cfg.verbose == True:
        print("Inizialization Done without problemas")
        print("Starting the training proces with the following configuration: ")
        bprint(dict(cfg_setup.config)) 
    
    ## ^  



    for epoch in tqdm.tqdm(range(epochs), desc="Training", ascii=True):
        batch_step(loader=train_loader, graph_structure=core_graph, Line_Encoder=Line_Encoder,
                   Edge_Positional_Encoder=Edge_Positional_Encoder,
                   Gnn_Encoder=Gnn_Encoder,
                   optimizer=optimizer,
                   criterion=criterion,
                   image_reshape=(height, width),
                   entities=list(pk.values()),
                   epoch=epoch,
                   mode="train")
        
        wandb.finish()
        exit()
        
        if epoch+1 % 10 == 0:
            batch_step(loader=val_loader, graph_structure=core_graph, Line_Encoder=Line_Encoder,
                   Edge_Positional_Encoder=Edge_Positional_Encoder,
                   Gnn_Encoder=Gnn_Encoder,
                   optimizer=optimizer,
                   criterion=criterion,
                   image_reshape=(height, width),
                   entities=list(pk.values()),
                   epoch=epoch,
                   mode="validation")
             


    batch_step(loader=test_loader, graph_structure=core_graph, Line_Encoder=Line_Encoder,
            Edge_Positional_Encoder=Edge_Positional_Encoder,
            Gnn_Encoder=Gnn_Encoder,
            optimizer=optimizer,
            criterion=criterion,
            image_reshape=(height, width),
            entities=list(pk.values()),
            epoch=epoch,
            mode="test")         
    
    wandb.finish()

        
        
if __name__ == "__main__":

    main()
        
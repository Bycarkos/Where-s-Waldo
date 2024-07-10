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


## Configuration Package
import hydra
from omegaconf import DictConfig, OmegaConf


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



def main():
    
    volumes = pipes.load_volumes()
    
    #  ^ Hydra thing
    #! The utility of this dictionary is to relate the groundtruth with the visual information
    pk = {"Noms_harmo":"nom", "cognom1_harmo":"cognom_1", "cognom2_harmo":"cognom_2", "parentesc_har":"parentesc", "ocupacio":"ocupacio"}
    batch_size = 128
    shuffle = True
    embedding_size = 128
    patch_size = 16
    # ^ 
    
    #? Define the dataset
    Graph = Graphset(Volumes=volumes, auxiliar_entities_pk=pk, add_edge_features=True)
    Graph._initialize_nodes(embedding_size = embedding_size)
    Graph._initialize_image_information()
    Graph._initialize_edges()
    print(list(pk.values()))
    ##? Defining the collator
    data_loader = FamilyCollator(graph=Graph, batch_size=batch_size, shuffle=shuffle)
    
    
    ## ** model things
    core_graph = Graph._graph
    x_dict = core_graph.x_dict

    min_width = core_graph["mean_width"]
    max_heigh = core_graph["min_height"] + core_graph["mean_height"]
    
    width, height = utils.extract_optimal_shape(min_width=min_width, max_height=max_heigh, patch_size=patch_size) 
    width = int(width.item())
    height = int(height.item())
    
    ratio = width /height
    
    kernel_height = int((height*0.7))
    kernel_width = int(kernel_height * ratio)
    
    ## ** 
    
    ## ^ Model 

    Line_Encoder = cnn.LineFeatureExtractor(min_height=kernel_height, max_width=kernel_width, hidden_channels=64, output_channels=128, num_hidden_convolutions=3).to("cuda")
    Edge_Positional_Encoder = cnn.EdgeAttFeatureExtractor(min_height=kernel_height, max_width=kernel_width, hidden_channels=64, output_channels=128, number_of_entities=len(pk.values()), edge_embedding_size=128).to("cuda")
    Gnn_Encoder = gnn.HeteroGNN(hidden_channels=128, attributes=list(pk.values())).to("cuda")

    params = list(Line_Encoder.parameters()) + list(Edge_Positional_Encoder.parameters()) + list(Gnn_Encoder.parameters())
    
    optimizer = Adam(params=params, lr=1e-4)
    criterion = utils.contrastive_loss
    
    Line_Encoder.train()
    Edge_Positional_Encoder.train()
    Gnn_Encoder.train()
    
    ## ^

    for idx, (edge_index_dict, negative_edge_index_dict, population) in (enumerate(data_loader)):
        print(negative_edge_index_dict.keys())
        images_to_keep = []
        for individual_index in population:
            image = core_graph["image_lines"][individual_index].image()
            image = transforms.to_tensor(image)
            _, h, w = image.shape

            if w < width:
                image = transforms.resize(img=image, size=(height, width))
                
            else:
                image = image[:, :, :width]
                image = transforms.resize(img=image, size=(height, width))
            
            images_to_keep.append(image)


        batch = torch.from_numpy(np.array(images_to_keep)).to("cuda")
        population = population.to("cuda")
        x_dict = {key: value.cuda() for key, value in x_dict.items()}
        edge_index_dict = {key: value.cuda() for key, value in edge_index_dict.items()}
        
        ## Extract visual information
        individual_features = Line_Encoder(x=batch)
        edge_features = Edge_Positional_Encoder(x=batch)
        
        # update the information of the individuals 
        x_dict["individuals"][population] = individual_features
        
        features_dict = Gnn_Encoder(x_dict, edge_index_dict, edge_features, population)

        ### *Task loss
        loss = 0
        
        for attribute in list(pk.values()):
            edge_similar_name = (attribute, "similar", attribute)
            edge_index_similar = edge_index_dict[edge_similar_name].to("cuda")
            
            
            positive_labels = torch.ones(edge_index_similar.shape[1])
            
            negative_edge_index_similar = negative_edge_index_dict[edge_similar_name].to("cuda")
            negative_labels = torch.zeros(negative_edge_index_similar.shape[1])
            
            gt = torch.cat((positive_labels, negative_labels), dim=0).to("cuda")
            edge_index = torch.cat((edge_index_similar, negative_edge_index_similar), dim=1)
            x1 = x_dict[attribute][edge_index[0,:]]
            x2 = x_dict[attribute][edge_index[1, :]]
            loss += criterion(x1, x2, gt)
                
        
        exit()
        
        
if __name__ == "__main__":

    main()
        
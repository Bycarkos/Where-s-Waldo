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
               model: Type[nn.Module],
               optimizer: Type[torch.optim.Adam],
               criterion,
               image_reshape: Tuple[int, int], 
               entities: Tuple[str, ...],
               epoch: int, 
               mode:str="train"):

        height, width = image_reshape
        epoch_train_loss = 0
        epoch_entities_losses = {key+"_"+mode+"_loss":0 for key in entities}
        #with torch.autograd.set_detect_anomaly(True):
        for idx, (edge_index_dict, negative_edge_index_dict, population) in tqdm.tqdm(enumerate(loader), ascii=True):
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

            ## Send information to the GPU
            batch = torch.from_numpy(np.array(images_to_keep)).to(device=device)
            population = population.to(device=device)
            x_dict = {key: value.to(device) for key, value in x_dict.items()}
            edge_index_dict = {key: value.to(device) for key, value in edge_index_dict.items()}
            
            ## Extract visual information
            individual_features = model.encode_visual_information(x=batch)
            edge_features = model.encode_edge_positional_information(x=batch)            
            x_dict["individuals"][population] = individual_features  # update the information of the individuals 

            
            x_dict = model.update_embeddings_with_message_passing(x=x_dict, edge_index=edge_index_dict, edge_attributes=edge_features, population=population) #message_passing
            #x_dict.update(features_dict)

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
                loss += criterion(x1, x2, gt)
                batch_entites_loss[attribute+"_"+mode+"_loss"] += loss

            loss.backward()
            optimizer.step()
            
            
        
        epoch_train_loss += loss
        epoch_entities_losses.update(batch_entites_loss)
        epoch_entities_losses[mode+"_loss"] = epoch_train_loss

        print(f"Epoch {epoch}: Loss: {epoch_train_loss}")

        wandb.log(epoch_entities_losses)

        return epoch_train_loss
    









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
    # ^ 
    
    #? Define the dataset
    Graph = Graphset(Volumes=volumes, auxiliar_entities_pk=pk, add_edge_features=True)
    Graph._initialize_nodes(embedding_size = embedding_size)
    Graph._initialize_image_information()
    Graph._initialize_edges()
    ##? 

    #? Define the Collator
    collator_train = FamilyCollator(graph=Graph, batch_size=batch_size, shuffle=shuffle, mode="train")
    collator_validation = FamilyCollator(graph=Graph, batch_size=batch_size, shuffle=shuffle, mode="validation")
    collator_test = FamilyCollator(graph=Graph, batch_size=batch_size, shuffle=shuffle, mode="test")



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
    kernel_width = int(kernel_height * (1-ratio_kernel))

    ## ** 
    
    ## ^ Model 
    cfg_model.line_encoder.kernel_height = kernel_height
    cfg_model.line_encoder.kernel_width = kernel_width
    cfg_model.gnn_encoder.attributes = list(pk.values())


    model = MMGCM(visual_encoder=cnn.LineFeatureExtractor, gnn_encoder=gnn.HeteroGNN, edge_encoder=cnn.EdgeAttFeatureExtractor, cfg=cfg_model).to(device)

    
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

        train_loss  = batch_step(loader=collator_train, graph_structure=core_graph, 
                                    model=model, 
                                    criterion=criterion, 
                                    optimizer=optimizer,
                                    image_reshape=(height, width), 
                                    entities=list(pk.values()), 
                                    epoch=epoch, 
                                    mode="train")
        
        
        torch.cuda.empty_cache()


        
        if epoch+1 % 10 == 0:
            validation_loss = batch_step(loader=collator_validation, graph_structure=core_graph, 
                                    model=model, 
                                    criterion=criterion, 
                                    optimizer=optimizer,
                                    image_reshape=(height, width), 
                                    entities=list(pk.values()), 
                                    epoch=epoch, 
                                    mode="validation")
            
            if validation_loss < optimal_loss:
                os.makedirs("./checkpoints", exist_ok=True)
                dataset_name = cfg_data.dataset.name
                model_name = f"./checkpoints/{dataset_name}+_mmgcm.pt"

                torch.save(model.state_dict(), model_name)

             


    test_loss = batch_step(loader=collator_test, graph_structure=core_graph, 
                                    model=model, 
                                    criterion=criterion,
                                    optimizer=optimizer, 
                                    image_reshape=(height, width), 
                                    entities=list(pk.values()), 
                                    epoch=epoch, 
                                    mode="test")         
    
    if test_loss < optimal_loss:
        os.makedirs("./checkpoints", exist_ok=True)
        dataset_name = cfg_data.dataset.name
        model_name = f"./checkpoints/{dataset_name}+_mmgcm.pt"

        torch.save(model.state_dict(), model_name)
    
    wandb.finish()

        
        
if __name__ == "__main__":

    main()
        
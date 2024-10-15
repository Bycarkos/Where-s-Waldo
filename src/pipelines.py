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
    
    
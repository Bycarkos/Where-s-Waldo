

## Model Things
from models import gnn_encoders as gnn
from models import visual_encoders as cnn
from models.graph_construction_model import MMGCM


import utils
import visualizations as visu

## Common packages
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData
import torch_geometric.utils as tutils
from pytorch_metric_learning import miners, losses



## Typing Packages
from typing import *
from jaxtyping import Float, Array


## Common packages
import copy
from networkx.algorithms import bipartite
import numpy as np
import wandb
import tqdm



device = "cuda" if torch.cuda.is_available() else "cpu"
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
             entities:list):


    
    image_embeddings = []
    attribute_embeddings = []
    individual_entity_embeddings = []
    population_final = []
    
    for idx, dict_images in tqdm.tqdm(enumerate(loader), desc="Validation/Test Extracting Embeddings", leave=True, position=1):
        images = dict_images["image_lines"].to(device)

        population = dict_images["population"].to(device)
        

        image_embedding = model.encode_visual_information(images)
        
        image_embeddings.append(image_embedding)
        
        attribute_representation, individual_embeddings = model(x=images)
        
        attribute_embeddings.append(attribute_representation)
        individual_entity_embeddings.append(individual_embeddings)
    
        population_final.append(population)
    
    population_final = torch.cat(population_final, dim=-1)
    ## Extracting information
    attribute_embeddings = torch.cat(attribute_embeddings, dim=0)
    individual_entity_embeddings = torch.cat(individual_entity_embeddings, dim=0)
    visual_embeddings = torch.cat(image_embeddings, dim=0)
    
    edges_to_keep = [(attribute, "similar", attribute) for attribute in entities]
    subgraph = utils.extract_subgraph(graph=graph, population=population_final, edges_to_extract=edges_to_keep)
    loss = 0
    
    for idx, attribute in  tqdm.tqdm(enumerate(entities), desc="Computing Loss of Validation/Test", leave=True, position=2):
        

        edge_similar_name = (attribute, "similar", attribute)
        similar_population = subgraph[edge_similar_name].flatten().unique().to(device)                
        labels = torch.isin(population_final, similar_population).type(torch.int32)
        
        if attribute == "individual":
            embeddings = individual_entity_embeddings
            
        else:
            embeddings = attribute_embeddings[:, idx,:]
            
        loss_attribute = criterion(embeddings, labels)
        loss += (loss_attribute)
    

    print("VALIDATION LOSS: ", loss)
    
    
    population_final = population_final.to("cpu")
        ## Update the information in the graph
    graph.x_attributes[population_final] = attribute_embeddings.to("cpu")
    graph.x_image_entity = visual_embeddings.to("cpu")   
    graph.x_entity[population_final] = individual_entity_embeddings.to("cpu")
    
    exit()
    return loss







def batch_step(loader, 
               graph:Type[HeteroData], 
               model: Type[nn.Module],
               optimizer: Type[torch.optim.Adam],
               criterion: Type[nn.Module],
               entities: Tuple[str, ...],
               scheduler,
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
        
        epoch_train_loss = 0
        epoch_entities_losses = {key+"_train_loss":0 for key in entities}

        if language_distillation: 
            x_ocr = torch.from_numpy(np.array(graph.x_language)).to(device)

        
        for idx, dict_images in tqdm.tqdm(enumerate(loader), ascii=True):
            optimizer.zero_grad()

            images = dict_images["image_lines"].to(device)
            #ocr_indexes = dict_images["ocrs"].to(device)
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
                # Select embeddings based on the attribute
                if attribute == "individual":
                    embeddings = individual_embeddings
                else:
                    embeddings = attribute_representation[:, idx, :]

                    
                # Only compute language distillation if enabled
                if language_distillation:
                    ocr_indexes = dict_images["ocrs"].to(device)
                    selected_language_embeddings_idx = ocr_indexes[:, idx].to(torch.int32)
                    language_embeddings = x_ocr[selected_language_embeddings_idx, :]
                    
                    # Concatenate embeddings and labels in a single operation
                    labels = torch.cat((labels, labels), dim=0)
                    embeddings = torch.cat((embeddings, language_embeddings), dim=0)

                loss_attribute = criterion(embeddings, labels)
                loss += (loss_attribute) # + contrastive_loss_attribute) #distilattion_loss
                
                batch_entites_loss[attribute+"_train_loss"] += loss_attribute

            loss.backward()
            optimizer.step()
            scheduler.step()
            
            
        
        epoch_train_loss += loss
        epoch_entities_losses.update(batch_entites_loss)
        epoch_entities_losses["train_loss"] = epoch_train_loss

        print(f"Epoch {epoch}: Loss: {epoch_train_loss}")

        wandb.log(epoch_entities_losses)
        
        
        return loss
    
    
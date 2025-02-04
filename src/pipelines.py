

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
from  torchtyping import TensorType
 

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
             entities:list,
             epoch:int):


    
    image_embeddings = []
    attribute_embeddings = []
    individual_entity_embeddings = []
    population_final = []
    epoch_entities_losses = {attribute+"_validation_loss":0 for attribute in entities}
   
    for idx, dict_images in tqdm.tqdm(enumerate(loader), desc="Validation/Test Extracting Embeddings", leave=True, position=1):
        images = dict_images["image_lines"].to(device)

        population = dict_images["population"].to(device)
        

        
        
        attribute_representation, individual_embeddings, image_embedding = model(x=images)
        
        attribute_embeddings.append(attribute_representation)
        individual_entity_embeddings.append(individual_embeddings)
        image_embeddings.append(image_embedding)
    
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
    
        epoch_entities_losses[attribute+"_validation_loss"] += loss_attribute

    print("VALIDATION LOSS: ", loss)
    
    
    population_final = population_final.to("cpu")
        ## Update the information in the graph
    graph.x_attributes[population_final] = attribute_embeddings.to("cpu")
    graph.x_image_entity[population_final] = visual_embeddings.to("cpu")   
    graph.x_entity[population_final] = individual_entity_embeddings.to("cpu")

    metrics_R_precission, general_metrics = updload_r_precission(graph=graph)
    
    wandb.log(epoch_entities_losses)
    wandb.log(general_metrics)


    return loss



@torch.no_grad()
def updload_r_precission(graph: Type[HeteroData]) -> Dict:
    
    metrics_R_precission = {

    }

    general_metrics = {
            
        }
    computed_freq_name = {name: len(values) for name, values in graph["nom"].map_attribute_index.items() if len(values) > 10}
    computed_freq_surname = {name: len(values) for name, values in graph["cognom_1"].map_attribute_index.items() if len(values) > 10}
    computed_freq_ssurname = {name: len(values) for name, values in graph["cognom_2"].map_attribute_index.items() if len(values) > 10}

    name_freq  = sorted(list(computed_freq_name.items()), key= lambda k: (k[1]), reverse=True)
    surname_freq  = sorted(list(computed_freq_surname.items()), key= lambda k: (k[1]), reverse=True)
    second_surnname_freq  = sorted(list(computed_freq_ssurname.items()), key= lambda k: (k[1]), reverse=True)


    attribute_embeddings = graph.x_attributes

    
    name_high_freq, name_low_freq = [i[0] for i in name_freq if i[1] >= 100],  [i[0] for i in name_freq if i[1] <= 20] 
    surname_high_freq, surname_low_freq = [i[0] for i in surname_freq if i[1] >= 100],  [i[0] for i in surname_freq if i[1] <= 20] 
    ssurname_high_freq, ssurname_low_freq = [i[0] for i in second_surnname_freq if i[1] >= 100],  [i[0] for i in second_surnname_freq if i[1] <= 20] 

    for (att_idx, att, high_freq, low_freq) in [(0, "nom", name_high_freq, name_low_freq), (1, "cognom_1", surname_high_freq, surname_low_freq), (2, "cognom_2", ssurname_high_freq, ssurname_low_freq)]:
        metrics_R_precission[att] = {}
        att_mean_low = 0
        att_mean_high = 0
        att_mean = 0
        count_high = 0
        count_low = 0
        nn, distances = utils.extract_intra_cluster_nearest_neighboors_at_k(attribute_embeddings[:,att_idx], top_k=attribute_embeddings.shape[0]) # nn = TensorType["Nnodes", "Nnodes"]

        for ind_idx in range(attribute_embeddings.shape[0]):

            try:
                content_attribute = graph[att].map_index_attribute[ind_idx]
                
            except:
                continue
            
            
            relevant_individuals = graph[att].map_attribute_index[content_attribute]


            value = utils.r_precision(relevant_documents=relevant_individuals, retrieved_documents=nn[ind_idx])

            att_mean += value
            
            if content_attribute in high_freq:
                count_high += 1
                att_mean_high += value           
            
            elif content_attribute in low_freq:
                count_low += 1
                att_mean_low += value           

            metrics_R_precission[att][f"{att}/{content_attribute}"] = value

        general_metrics[f"high_mean_recall/{att}"] = att_mean_high/(count_high)
        general_metrics[f"low_mean_recall/{att}"] = att_mean_low/(count_low)
        general_metrics[f"mean_recall/{att}"] = att_mean/(attribute_embeddings.shape[0])

    return metrics_R_precission, general_metrics


def batch_step(loader, 
               graph:Type[HeteroData], 
               model: Type[nn.Module],
               optimizer: Type[torch.optim.Adam],
               criterion: Type[nn.Module],
               entities: Tuple[str, ...],
               scheduler,
               epoch: int):

        

        model.train()
        
        epoch_train_loss = 0
        epoch_entities_losses = {key+"_train_loss":0 for key in entities}
        population_final = []
        attribute_embeddings = []
        image_embeddings = []
        individual_embeddings = []


        for idx, dict_images in tqdm.tqdm(enumerate(loader), desc="Batch Loop", leave=True, position=1):
            optimizer.zero_grad()

            images = dict_images["image_lines"].to(device)
            population = dict_images["population"].to(device)

            attribute_representation, individual_representation, image_features = model(x=images)

            attribute_embeddings.append(attribute_representation)
            image_embeddings.append(image_features)
            individual_embeddings.append(individual_representation)  
            population_final.append(population)


            # ** Task loss
            loss = 0
            edges_to_keep = [(attribute, "similar", attribute) for attribute in entities]
            subgraph = utils.extract_subgraph(graph=graph, population=population, edges_to_extract=edges_to_keep)


            for idx, attribute in enumerate(entities):
        
                edge_similar_name = (attribute, "similar", attribute)
                similar_population = subgraph[edge_similar_name].flatten().unique().to(device)                
                
                labels = torch.isin(population, similar_population).type(torch.int32)
                # Select embeddings based on the attribute
                if attribute == "individual":
                    embeddings = individual_representation
                else:
                    embeddings = attribute_representation[:, idx, :]

                loss_attribute = criterion(embeddings, labels)
                loss += (loss_attribute) # + contrastive_loss_attribute) #distilattion_loss
                
                
                epoch_entities_losses[attribute+"_train_loss"] += loss_attribute


            epoch_train_loss += loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            
        population_final = torch.cat(population_final, dim=-1)
        attribute_embeddings = torch.cat(attribute_embeddings, dim=0)
        image_embeddings = torch.cat(image_embeddings, dim=0)
        individual_embeddings = torch.cat(individual_embeddings, dim=0)

        ## ?¿ Updating the graph information
        population_final = population_final.to("cpu")
        graph.x_attributes[population_final] = attribute_embeddings.to("cpu")
        graph.x_image_entity[population_final] = image_embeddings.to("cpu")   
        graph.x_entity[population_final] = individual_embeddings.to("cpu")

        epoch_train_loss /= len(loader)
        
        for k in epoch_entities_losses.keys():
            epoch_entities_losses[k] = epoch_entities_losses[k] / len(loader) 

        epoch_entities_losses["train_loss"] = epoch_train_loss

        print(f"Epoch {epoch}: Loss: {epoch_train_loss}")

        wandb.log(epoch_entities_losses)
        
        
        return epoch_train_loss
    
    
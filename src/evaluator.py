## Dataset Things

from data.volumes import Volume, Page, Line
from data.graphset import Graphset
from data.graph_sampler import GraphSampler, AttributeSampler
from src.data.ced_dataset import ImageDataset


import torch_geometric.utils as tutils


import data.volumes as dv

## Model Things
from models import gnn_encoders as gnn
from models import visual_encoders as cnn
from models.graph_construction_model import MMGCM
from models import edge_visual_encoders as EVE



### Utils
import utils 
import visualizations as visu


## Pipelines
import pipelines as pipes

## tasks
from tasks import record_linkage as rl

## Common packages
import torch.nn as nn
import torchvision.transforms.functional as transforms
import torch

from torch.optim import Adam

import numpy as np

## Typing Packages
from typing import *
from torch.utils.data import DataLoader



## Configuration Package
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

## Experiment Tracking packages
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



import umap
import umap.plot

device = "cuda" if torch.cuda.is_available() else "cpu"
import pprint
from beeprint import pp as bprint


from sklearn.model_selection import StratifiedKFold

import pdb

@hydra.main(config_path="./configs", config_name="eval", version_base="1.3.2")
def main(cfg: DictConfig):
    
    CFG_MODELS = cfg.models
    CFG_DATA = cfg.data
    CFG_SETUP    = cfg.setup


    evaluate_path = "evaluation/"

    
    
    #! The utility of this dictionary is to relate the groundtruth with the visual information AD-HOC
    pk = {"Noms_harmo":"nom", "cognom1_harmo":"cognom_1", "cognom2_harmo":"cognom_2", "parentesc_har":"parentesc", "ocupacio":"ocupacio"}
    
    #  ^ Hydra things
 
    batch_size = CFG_DATA.collator.batch_size
    shuffle = CFG_DATA.collator.shuffle
    number_volum_years = len(CFG_DATA.dataset.volumes) 
    checkpoint_name = CFG_MODELS.name_checkpoint
    embedding_size = CFG_MODELS.common_configuration.embedding_size

    # ^ 
    
    ## & Extract the dataset and the information in this case 
    volumes = pipes.load_volumes(cfg=CFG_DATA) 
    image_dataset = ImageDataset(Volumes=volumes, cfg=CFG_DATA.dataset)
    df_transcriptions = image_dataset._total_gt
    n_different_individuals = image_dataset._total_individual_nodes
    graphset = Graphset(total_nodes=n_different_individuals,
                        df_transcriptions=df_transcriptions,
                        n_volumes=len(volumes),
                        graph_configuration=CFG_DATA.graph_configuration,
                        auxiliar_entities_pk = pk)

    sampler = AttributeSampler(graph=graphset._graph, batch_size=batch_size, shuffle=shuffle)


    print("Generating DataLoader")

    total_loader = DataLoader(dataset=image_dataset, 
                            batch_size = batch_size,
                            collate_fn=image_dataset.collate_fn,
                            num_workers=0,
                            shuffle=True,
                            pin_memory=True)

    print("DATA LOADED SUCCESFULLY")
    H = image_dataset.general_height 
    W = image_dataset.general_width // 16

    CFG_MODELS.edge_visual_encoder.input_atention_mechanism = H*W

    model = MMGCM(visual_encoder=cnn.LineFeatureExtractor, gnn_encoder=gnn.AttributeGNN, edge_encoder=EVE.EdgeAttFeatureExtractor, cfg=CFG_MODELS).to(device)
    model_name = f"./checkpoints/{checkpoint_name}.pt"
    name_embeddings = f"{checkpoint_name}"

    model.load_state_dict(torch.load(model_name))
    model.to(device)
    print("MODEL LOADED SUCCESFULLY")

    
    core_graph = graphset._graph
    
    if CFG_MODELS.add_language is True:
        
        filepath  = f"embeddings/language_embeddings_{number_volum_years}_{embedding_size}_entities_{len(CFG_SETUP.configuration.compute_loss_on)}.pkl"
        
        if os.path.exists(filepath):
            language_embeddings = utils.read_pickle(filepath)
                
        else:
            language_embeddings = utils.extract_language_embeddings(list(image_dataset._map_ocr.keys()), embedding_size=embedding_size, filepath=filepath)
    
        print("LANGUAGE EMBEDDINGS EXTRACTED SUCCESFULLY")
    
        core_graph.x_language = language_embeddings

        print(image_dataset._map_ocr)
        
    core_graph.epoch_populations = image_dataset._population_per_volume

    

    if os.path.exists(f"embeddings/{name_embeddings}.pkl"):
        attribute_embeddings = utils.read_pickle(filepath=f"embeddings/{name_embeddings}.pkl")
        attribute_embeddings = attribute_embeddings.cpu()
    else:            
        attribute_embeddings, individual_embeddings = utils.extract_embeddings_from_graph(loader=total_loader,
                                              graph=core_graph,
                                              model=model)
        

        attribute_embeddings = attribute_embeddings.cpu()
        individual_embeddings = individual_embeddings.cpu()
                        
        utils.write_pickle(info=attribute_embeddings, 
                           filepath=f"embeddings/{name_embeddings}.pkl")
        
#    print(model._edge_encoder.attention_values)
    exit()
    path_save_plots = os.path.join(evaluate_path, checkpoint_name)
    
    visu.plot_attribute_metric_space(attribute_embedding_space=attribute_embeddings.cpu(),
                                     fig_name=os.path.join(path_save_plots, "Attributes_Distribution.jpg"))
    
    
    pos_embedding = model._edge_encoder.pos_emb[:1000].detach().cpu().numpy().reshape(1, -1)
    
    #visu.plot_positional_encoding(positional_encoding=pos_embedding, fig_name=os.path.join(path_save_plots, "Positional_embedding.jpg"))
    #exit()
        
    ## ** TEST EXTRACTION
    
    candidate_pairs = core_graph[("individual", "same_as", "individual")].negative_sampling
    true_pairs = core_graph[("individual", "same_as", "individual")].edge_index
    
    # extract the population from the last time
    final_time_gap_population = list(core_graph.epoch_populations[-2]) + list(core_graph.epoch_populations[-1]) 
    final_time_gap_population = torch.tensor(final_time_gap_population).type(torch.int64)
    
    ## Extract candidate pairs 
    #specific_subgraph_candidate, _ = tutils.subgraph(subset=final_time_gap_population, edge_index=candidate_pairs[:2,:].type(torch.int64))
    ## Extract true pairs for the last epochs
    mask = torch.isin(true_pairs[:2, :], final_time_gap_population).all(dim=0)
    mask_candidate = torch.isin(candidate_pairs[:2,:], final_time_gap_population).all(dim=0)
    ## /& This one is used to get the final numbers ( This is train true pairs)
    specific_true_pairs_subgraph = true_pairs[:, mask]
    specific_subgraph_candidate = candidate_pairs[:, mask_candidate]
    
    ### Fins aquí tot bé
    
    ## & TEST SAME AS
    X_test_indexes = torch.cat((specific_true_pairs_subgraph, specific_subgraph_candidate), dim=1).type(torch.int32).numpy()
    X_test = (attribute_embeddings[X_test_indexes[0], :] - attribute_embeddings[X_test_indexes[1],:]).pow(2).sum(-1).sqrt()
    y_test = X_test_indexes[-1]
    
    
    ## ^ TRAIN EXTRACTION 
    earlies_time_populations = []
    for pop in core_graph.epoch_populations[:-1]:
        earlies_time_populations += list(pop)
        
        
    earlies_time_populations = torch.tensor(earlies_time_populations).type(torch.int64)

    mask = torch.isin(true_pairs[:2, :], earlies_time_populations).all(dim=0)
    mask_candidate_train = torch.isin(candidate_pairs[:2,:], earlies_time_populations).all(dim=0)
    
    specific_true_pairs_subgraph_train = true_pairs[:, mask]
    specific_subgraph_candidate_train = candidate_pairs[:, mask_candidate_train]

    X_train_indexes = torch.cat((specific_true_pairs_subgraph_train, specific_subgraph_candidate_train), dim=1).numpy()
    X_train = (attribute_embeddings[X_train_indexes[0], :] - attribute_embeddings[X_train_indexes[1],:]).pow(2).sum(-1).sqrt()

    y_train = X_train_indexes[-1] #torch.cat((torch.ones((specific_true_pairs_subgraph_train.shape[1])), torch.zeros((specific_subgraph_candidate_train.shape[1]))), dim=0).numpy()

    metrics_rl = rl.record_linkage_with_logistic_regression(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, candidate_sa=X_test_indexes,
                                                                                                            random_state=0, penalty="l2", solver="newton-cholesky", class_weight="balanced", n_jobs=-1)
    

    os.makedirs(os.path.join("KNN/", checkpoint_name), exist_ok=True)

    dict_nn = utils.extract_intra_cluster_nearest_neighboors_at_k(attribute_embeddings, top_k=500)


    nn, distances = dict_nn
    pdb.set_trace()

    utils.write_pickle(dict_nn, filepath=os.path.join("KNN/", checkpoint_name, "NN.pkl"))
    
        
    
        
if __name__ == "__main__":

    main()
        
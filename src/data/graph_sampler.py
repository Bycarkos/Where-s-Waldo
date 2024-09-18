
## Custom packages
from data.volumes import Volume, Page, Line, sort_key
from data.graphset import Graphset
from torch.utils.data import Sampler

## geometrical data packages
import torch
import torchvision.transforms.functional as T
from torch_geometric.data import HeteroData
from torch.utils.data import Dataset, DataLoader

## Typing packages
from torchtyping import TensorType, patch_typeguard
from typing import *
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

## commmon packages
import pandas as pd
import os
import glob
from PIL import Image
import itertools
import re
import pdb
import random
from omegaconf import DictConfig, OmegaConf

    


class GraphSampler():
    
    def __init__(self, graph: Type[HeteroData], 
                        batch_size:int=64, 
                        shuffle:bool=True, mode:str="train") -> None:
        
        """
        Initializes the FamilyCollator with a graph, batch size, partition ratios, and shuffle option.

        Args:
            graph (Type[Graphset]): An instance of the Graphset class.
            batch_size (int): Size of each batch. Default is 64.
            partition_ration (Tuple[float, ...]): Ratios for train, validation, and test partitions. Default is (0.8, 0.1, 0.1).
            shuffle (bool): Whether to shuffle the data. Default is True.
        """
        
        self._graph = graph
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._mode = mode
        #extract type of nodes
       
    def _change_mode(self, mode:str="validation"):
        self._mode = mode
               

class AttributeSampler(GraphSampler):

    def __init__(self, graph: Type[HeteroData], batch_size:int=64, shuffle:bool=True, partition_ration: Tuple[float, ...] = [0.8, 0.1, 0.1]):

        

        super().__init__(graph=graph, batch_size=batch_size,  shuffle=shuffle)

        self._partition_ratio = partition_ration
        self._total_population = self._graph.x_attributes.shape[0]

        self._train_ratio, self._validation_ratio, self._test_ratio = partition_ration
        self.__create_partition()

    def __create_partition(self):
        

        population = torch.arange(self._total_population)
       
        index_permutation = torch.randperm(self._total_population)
        train_partition_families  = index_permutation[:int(index_permutation.shape[0]*self._train_ratio)]
        validation_partition_families = index_permutation[int(index_permutation.shape[0]*self._train_ratio): int(index_permutation.shape[0]*(self._train_ratio+ self._validation_ratio))]
        test_partition_families = index_permutation[int(index_permutation.shape[0]*(self._train_ratio+ self._validation_ratio)):]


        self._train_population = population[train_partition_families]
        self._validation_population = population[validation_partition_families]
        self._test_population = population[test_partition_families]



    def _generate_m_space_batch(self, x_embeddings: TensorType["n_attributes", "num_nodes", "embedding_size"],
                                    attributes:list,
                                    top_k:int=10, 
                                    mirro_sampling:int=20) -> None:
        
        self._batch = [[] for i in range(self._batch_size)]

        for idx, att in enumerate(attributes):

            embeddings = x_embeddings[idx]
            distances = torch.cdist(embeddings, embeddings, p=2)
            distances.fill_diagonal_(float('inf'))
            top_k_values, top_k_indices = torch.topk(distances, top_k, dim=1, largest=False)
            nearest_neigbors = top_k_indices[self._train_population]

            for idx, individual in enumerate(self._train_population):
                self._batch[idx % self._batch_size].extend(nearest_neigbors[idx].cpu().numpy())
                
                ## Extract the population with the same name of the individual
                try:
                    attrib_ind = self._graph[att].map_index_attribute[individual.item()]
                except Exception as e:
                    print("SOME ERROR AT SOME POINT IN THE COLLATOR, PROBABLY DUE TO A TRANSCRIPTION ERROR")
                    continue

                similar_names = self._graph[att].map_attribute_index[attrib_ind]
                #print(len(similar_names))
                    
                random_sampling = random.sample(list(similar_names), min(len(similar_names), mirro_sampling))
                set_indexes_attribute = torch.from_numpy(np.array(random_sampling))
                
                ## get rid of the population who is in evaluation time
                mask_individuals_train = torch.isin(set_indexes_attribute, self._train_population)
                nodes_similar_to_add = set_indexes_attribute[mask_individuals_train]
                
                ## add the new nodes if thei're not in the topk nn 
                self._batch[idx % self._batch_size].extend(nodes_similar_to_add.cpu().numpy())
                self._batch[idx % self._batch_size] = list(set(self._batch[idx % self._batch_size]))

        return iter(self._batch)
    def __iter__(self):
        return iter(self._batch)
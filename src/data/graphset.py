
## Custom packages
from data.volumes import Volume, Page, Line, sort_key


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
import tqdm
import pdb
from omegaconf import DictConfig, OmegaConf


    


class Graphset():
    

    def __init__(self, total_nodes:int,
                 df_transcriptions: Type[pd.DataFrame],
                 n_volumes: int,
                 graph_configuration: DictConfig,
                 auxiliar_entities_pk: Optional[Tuple[str, ...]] = None) -> None:
        

        
        ## information extracted from visual dataset
        self._total_individual_nodes = total_nodes
        self._total_gt = df_transcriptions
        self._nvolumes = n_volumes

        ## graph configuration
        self.attribute_type_of_nodes = graph_configuration.attribute_type_of_nodes
        self.entity_type_of_nodes = graph_configuration.entity_type_of_nodes
        self._type_of_nodes = self.attribute_type_of_nodes + self.entity_type_of_nodes
        
        
        self._type_of_attribute_edges = graph_configuration.attribute_edges
        self._type_of_entity_edges = graph_configuration.entity_edges
        self._node_embedding_size = graph_configuration.node_embedding_size
        self._edge_embedding_size = graph_configuration.edge_embedding_size
        
        
        #initialize core information such as the total of attributes, individuals, entites etc...
        self.__mapping_pk = auxiliar_entities_pk 



        ### Core Graph construction 
        self._graph = HeteroData()

        self._initialize_nodes()
        self._initialize_edges()

        
    def _initialize_nodes(self):
        
        """
        Initializes nodes in the graph with placeholders for each kind of node attribute and all attributes.
        """
        self._graph.x_attributes = torch.zeros(size=(self._total_individual_nodes, len(self.attribute_type_of_nodes), self._node_embedding_size), dtype=torch.float32)
        self._graph.x_entity = torch.zeros(size=(self._total_individual_nodes, len(self.entity_type_of_nodes), self._node_embedding_size), dtype=torch.float32)
        self._graph.map_attribute_nodes = {idx: node_type for idx, node_type in enumerate(self.attribute_type_of_nodes)}
        self._graph.map_entity_nodes = {idx: node_type for idx, node_type in enumerate(self.entity_type_of_nodes)}
        
            
            
    def _initialize_edges(self):
        self.__initialize_individual_attribute_edges() ## every line has the sma index name and surname because is a replication
        self.__initialize_attribute_edges(auxiliar_search_pk=self.__mapping_pk)
        self.__initialize_same_as_edges()
        self.__intitlialize_family_edges()



    def __initialize_individual_attribute_edges(self):
        
        """
        Initializes edges between individuals and attributes.

        Args:
            auxiliar_search_pk (Optional[Tuple[str, ...]]): Optional tuple of auxiliary entity primary keys.
        """

        ## placeholding
        total_nodes = torch.arange(self._total_individual_nodes)[None,:]

        for idx, etype in (enumerate(self._type_of_attribute_edges)):
            edge_key = ("individual", etype, etype)
            edge_index = torch.cat((total_nodes, total_nodes), dim=0)

            self._graph[edge_key].edge_index = edge_index
            self._graph[edge_key].negative_sampling = []
            self._graph[edge_key].edge_attr =  torch.zeros((self._edge_embedding_size), dtype=torch.float32)
   

    def __initialize_attribute_edges(self, auxiliar_search_pk:Optional[Dict[str, str]] = None):
        
        """
        Initializes attribute edges based on harmonized data.

        Args:
            auxiliar_search_pk (Optional[Dict[str, str]]): Optional dictionary of auxiliary search primary keys.
        """
        
        ## placeholding

        total_entities_pk = (self._type_of_attribute_edges) if auxiliar_search_pk == None else auxiliar_search_pk
        total_entities_pk = {harmo:etype   for harmo, etype in total_entities_pk.items() if etype in self.attribute_type_of_nodes}
        total_nodes = np.arange(self._total_individual_nodes)
        
        

        for idx, (harmo, etype) in (enumerate(total_entities_pk.items())):
            grouped_attributes = {}
            map_index_attribute = {}
            grouped_ns = {}

            ### Group by the harmonized data
            grouped = self._total_gt.groupby(harmo, as_index=False)
            edge_index = []


            for iddx, (group_name, _) in  tqdm.tqdm(enumerate(grouped), desc="Generating Adj for attributes"):
                index_group_by = grouped._get_index(group_name)
                grouped_attributes[group_name] = index_group_by
                map_index_attribute.update({ind:group_name for ind in index_group_by})

                negative_sampling_nodes_idx = np.isin(total_nodes, index_group_by)
                negative_sampling_nodes_idx = negative_sampling_nodes_idx != True
                negative_sampling = total_nodes[negative_sampling_nodes_idx]
                grouped_ns[group_name] = negative_sampling
                #negative_sampling = np.array(list(itertools.product(index_group_by, total_nodes[negative_sampling_nodes_idx])))
                if len(index_group_by) >1:
                    edge_index.extend(self.generate_index_permutations(grouped._get_index(group_name)))
                #ns_edge_index.extend(negative_sampling)
                
            edge_index = torch.tensor(edge_index).T
            edge_index = torch.cat((edge_index, torch.zeros((1, edge_index.shape[1])) + idx), dim=0)
            #ns_edge_index = torch.tensor(ns_edge_index).T
            
            ## select the non harmonized data
            self._graph[etype].map_attribute_index = grouped_attributes
            self._graph[etype].map_index_attribute = map_index_attribute
            
            self._graph[(etype, "similar", etype)].edge_index = edge_index.type(torch.int64)
            self._graph[(etype, "similar", etype)].negative_sampling = grouped_ns
            ## Extract all possible negative sampling for each individual
            
    def __initialize_same_as_edges(self):

        """
        Initializes 'same as' edges between individuals based on ground truth data.
        """

        true_pairs = []

        ## Filter as possible
        self._total_gt["id"] = self._total_gt["id"].fillna("SFLL")

        grouped = self._total_gt.groupby('id', as_index=False)
        count = 0
        for _, (group_name, group) in  tqdm.tqdm(enumerate(grouped), desc="Generating Adj for Same As"):
            total_same_as = grouped._get_index(group_name)

            if len(total_same_as) == 2: 
                count += 1 
                true_pairs.extend([total_same_as])
                
            elif (len(total_same_as) > 2) and (len(total_same_as) <= (self._nvolumes)):
                for i in range(1, len(total_same_as)):
                    count += 1
                    true_pairs.extend([np.array([total_same_as[i-1], total_same_as[i]])])

        true_pairs = torch.from_numpy(np.array(true_pairs)).T
        etype = ("individual", "similar", "individual")
        self._graph[etype].edge_index = torch.cat((true_pairs.unique(dim=1), torch.ones((1, true_pairs.unique(dim=1).shape[1]))), dim=0)
        
        candidate_pairs = self._graph[("cognom_2", "similar", "cognom_2")].edge_index

        ## ** Extract both candidate pairs and true pairs
        candidate_pairs = torch.cat([candidate_pairs[[0, 1]], true_pairs, true_pairs[[1,0]]], dim=1).unique(dim=1)

        inverse_true_pairs   = torch.cat((true_pairs, true_pairs[[1,0]] ), dim=1).unique(dim=1)
        gt_candidate_pairs = torch.zeros(candidate_pairs.size(1), dtype=torch.bool)
        
        batch_size = 100000  # Adjust the batch size depending on your memory constraints 
        for i in range(0, candidate_pairs.size(1), batch_size):
            batch = candidate_pairs[:, i:i+batch_size]
            # Check if each batch of candidate pairs exists in true_pairs_undirected
            gt_candidate_pairs[i:i+batch_size] = (batch.unsqueeze(2) == inverse_true_pairs.unsqueeze(1)).all(dim=0).any(dim=1)

        final_candidate_pairs = candidate_pairs[:, ~gt_candidate_pairs]
        # Assign the boolean result to y_candidate_pairs and create negative sampling

        self._graph[etype].negative_sampling = torch.cat((final_candidate_pairs, torch.zeros((1, final_candidate_pairs.shape[1]))), dim=0)


    def __intitlialize_family_edges(self):
        
        """
        Initializes family edges between individuals based on ground truth data.
        """

        families = []

        ## Filter as possible
        grouped = self._total_gt.groupby('id_llar_general', as_index=False)
        population_non_family_head = []
        map_ind_family = {}
        map_family_ind = {}
        for iidx, (group_name, group) in  tqdm.tqdm(enumerate(grouped), desc="Generating Adj for families"):
            #print(group_name)
            real_indexes = grouped._get_index(group_name)
            for ind in real_indexes:
                map_ind_family[ind] = iidx
                
            map_family_ind[iidx] = real_indexes
            #print(real_indexes)
            if len(real_indexes) > 1:
                parentesc = (group["parentesc"].values)
                list_pairs = self.generate_families(list(zip(parentesc, real_indexes)))

                    
                families.extend(list_pairs)

        families = torch.tensor(families).T
        
        etype = ("individual", "family", "individual")
        
        self._graph[etype].edge_index = families
        self._graph[etype].y = torch.ones(families.shape[1])
        self._graph[etype].negative_sampling = []
        self._graph[etype].non_head_population = population_non_family_head
        self._graph[etype].map_ind_family = map_ind_family
        self._graph[etype].map_family_ind = map_family_ind


    def generate_families(self, index_list):
        
        """
        Generates family edges from a list of indexes.

        Args:
            index_list (List[Tuple[str, int]]): List of tuples containing family roles and indexes.

        Returns:
            List[Tuple[int, int]]: List of family edges.
        """
        
        _, real_indexes = list(zip(*index_list))
        real_indexes = list(real_indexes)
        dic_families = dict(index_list)

        index_jefe = dic_families.get("jefe", None) if dic_families.get("jefe", None) is not None else dic_families.get("esposa", None)
        if index_jefe is not None:
            real_indexes.remove(index_jefe)

            return [(index_jefe, idx_pair) for idx_pair in real_indexes]
        else:
            return []

    @staticmethod
    def generate_index_permutations(index_list):
        """
        Generates permutations of indexes for creating edges.

        Args:
            index_list (List[int]): List of indexes.

        Returns:
            List[Tuple[int, int]]: List of index permutations.
        """
        if len(index_list) > 1:
            return [pair for pair in itertools.permutations(index_list, 2) if pair[0] != pair[1]]
        else:
            return []

    @staticmethod
    def contains_letters_and_numbers(s):
        """
        Checks if a string contains both letters and numbers.

        Args:
            s (str): Input string.

        Returns:
            bool: True if the string contains both letters and numbers, False otherwise.
        """
        
        return bool(re.search(r'[a-zA-Z]', s)) and bool(re.search(r'[0-9]', s))







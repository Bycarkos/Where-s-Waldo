## Custom Imports
from data.graphset import Graphset
from data.volumes import *

import torch

import torchvision.transforms.functional as transforms
import cv2 as cv



from typing import *
from torch.utils.data import Sampler
from torch_geometric.data.sampler import HeteroData

import pickle
import copy
import random 
import json
import glob
import numpy as np

import pdb


class GraphSampler(Sampler):
    
    def __init__(self, graph: Type[Graphset], batch_size:int=64, partition_ration: Tuple[float, ...] = [0.8, 0.1, 0.1], shuffle:bool=True, mode:str="train", image_reshape:Optional[Tuple[int, int]]=None) -> None:
        
        super(self, GraphSampler).__init__()
        """
        Initializes the FamilyCollator with a graph, batch size, partition ratios, and shuffle option.

        Args:
            graph (Type[Graphset]): An instance of the Graphset class.
            batch_size (int): Size of each batch. Default is 64.
            partition_ration (Tuple[float, ...]): Ratios for train, validation, and test partitions. Default is (0.8, 0.1, 0.1).
            shuffle (bool): Whether to shuffle the data. Default is True.
        """
        
        self._graphset = graph
        self._graph = graph._graph
        self._train_ratio, self._validation_ratio, self._test_ratio = partition_ration
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._image_reshape = image_reshape
        self._mode = mode
        #extract type of nodes
    
    
    
    def _change_mode(self, mode:str="validation"):
        self._mode = mode
        
        
    def extract_subgraph(self, population:List[Tuple[int, int]], edges_type:List[Tuple[str, str, str]]) -> Dict:
        
        """
        Extracts a subgraph based on the specified population of nodes and edge types.

        Args:
            population (List[Tuple[int, int]]): List of nodes to include in the subgraph.
            edges_type (List[Tuple[str, str, str]]): List of edge types to include in the subgraph.

        Returns:
            Dict: A dictionary containing the filtered adjacency lists for each edge type.
        """
        
        subgraph = {}
        negative_subgraph = {}
        
        
        for possible_edge in edges_type:
                
            adj = self._graph[possible_edge].edge_index
            negative_sampling = self._graph[possible_edge].negative_sampling

            indexes_to_retrieve = torch.isin(adj, population).all(dim=0)
            filtered_adj = adj[:, indexes_to_retrieve]
            
            if len(negative_sampling) != 0:
                negative_indexes_to_retrive = torch.isin(negative_sampling, population).all(dim=0)
                negative_subgraph[possible_edge] = negative_sampling[:, negative_indexes_to_retrive]

            subgraph[possible_edge] = filtered_adj
            
        
        return subgraph, negative_subgraph


        
class FamilyCollator(Collator):
    
    """
    FamilyCollator is responsible for partitioning a graph into train, validation, 
    and test sets, and for collating batches of subgraphs for these partitions.

    Attributes:
        _graphset (Graphset): An instance of the Graphset class.
        _graph (Graph): The graph structure extracted from the Graphset instance.
        _train_ratio (float): Ratio of the training set partition.
        _validation_ratio (float): Ratio of the validation set partition.
        _test_ratio (float): Ratio of the test set partition.
        _batch_size (int): Size of each batch.
        _shuffle (bool): Whether to shuffle the data.
        _graph_batch (List[List]): Batches of subgraphs.
        _same_as_graph (List[List]): Batches of 'same as' edges.

    Methods:
        __create_partition(): Creates partitions for training, validation, and test sets.
        collate_test(): Collates the test set subgraph.
        collate_validation(): Collates the validation set subgraph.
        extract_subgraph(population, edges_type): Extracts a subgraph based on nodes and edge types.
        __iter__(): Iterator to generate batches of subgraphs.
    """

    def __init__(self, graph: Type[Graphset], batch_size:int=64, partition_ration: Tuple[float, ...] = [0.8, 0.1, 0.1], shuffle:bool=True, mode:str="train", image_reshape:Optional[Tuple[int, int]]=None) -> None:
        
        """
        Initializes the FamilyCollator with a graph, batch size, partition ratios, and shuffle option.

        Args:
            graph (Type[Graphset]): An instance of the Graphset class.
            batch_size (int): Size of each batch. Default is 64.
            partition_ration (Tuple[float, ...]): Ratios for train, validation, and test partitions. Default is (0.8, 0.1, 0.1).
            shuffle (bool): Whether to shuffle the data. Default is True.
        """
        

        super().__init__(graph=graph, batch_size=batch_size, partition_ration=partition_ration, shuffle=shuffle, mode=mode, image_reshape=image_reshape)
        self._batch = [[] for i in range(batch_size)]

        self.__create_partition()
        
    def __create_partition(self):
        
        """
        Creates partitions for training, validation, and test sets based on 'same as' edges.
        """   
        
    ## ** Extract same as and get the families that each individual belongs 
        same_as_edge_index = self._graph[("individual", "same_as", "individual")].edge_index.unique(dim=1)        
        map_ind_family :dict = self._graph[("individual", "family", "individual")].map_ind_family
        
                
        mask_family_src =  torch.from_numpy(np.array([map_ind_family[i.item()] for i in same_as_edge_index[0,:]]))
        
    ## ** Extract the possible families id
        possible_families = torch.unique(mask_family_src)        

    ## ** Divide the failies by src because we keep the simetry on family heads    
        index_permutation = torch.randperm(possible_families.shape[0])
        train_partition_families  = index_permutation[:int(index_permutation.shape[0]*self._train_ratio)]
        validation_partition_families = index_permutation[int(index_permutation.shape[0]*self._train_ratio): int(index_permutation.shape[0]*(self._train_ratio+ self._validation_ratio))]
        test_partition_families = index_permutation[int(index_permutation.shape[0]*(self._train_ratio+ self._validation_ratio)):]

    ## ** Extract the possible families form the src (first epoch)
        mask_group_fam_train = possible_families[train_partition_families]
        mask_group_fam_val = possible_families[validation_partition_families]
        mask_group_fam_test = possible_families[test_partition_families]
        
         
    ## ** Mask the families by split 
        mask_same_as_train = torch.isin(mask_family_src, mask_group_fam_train)
        mask_same_as_val = torch.isin( mask_family_src, mask_group_fam_val)
        mask_same_as_test = torch.isin(mask_family_src, mask_group_fam_test)

    ## ** Keep the all the same as that belongs to the same family. This keeps the family on the first year, and we allow in the next epoch to split this family to generate new
        self._train_edge_index =  same_as_edge_index[:, mask_same_as_train]
        self._validation_edge_index = same_as_edge_index[:, mask_same_as_val]
        self._test_edge_index = same_as_edge_index[:, mask_same_as_test]
        
    ## ** Nodes partitions from same as
        self._ind_nodes_train_src, self._ind_nodes_train_trg = self._train_edge_index[0,:], self._train_edge_index[1,:]
        self._ind_nodes_validation_src, self._ind_nodes_validation_trg  = self._validation_edge_index[0,:], self._validation_edge_index[1,:]
        self._ind_nodes_test_src, self._ind_nodes_test_trg = self._test_edge_index[0,:], self._test_edge_index[1,:]
    

    def _change_mode(self, mode:str="validation"):
        self._mode = mode
          
    
    def __iter__(self):
        
        
        """
        Iterator to generate batches of subgraphs for training.

        Yields:
            Tuple[Dict, torch.Tensor]: A tuple containing a batch subgraph and the final nodes for each batch.
        """
        
        graph_batch = [[] for i in range(self._batch_size)]
        same_as_graph = copy.copy(graph_batch)

        if self._mode == "train":
            possible_population = self._train_edge_index
        elif self._mode == "validation":
            possible_population = self._validation_edge_index

        else:
            possible_population = self._test_edge_index 

        distribution = [ pair for pair in possible_population.numpy().T]
        random.shuffle(distribution)

        families = self._graph[("individual", "family", "individual")].edge_index
        possible_edges = (self._graph.edge_index_dict.keys())


        for idx, pair in enumerate(distribution):
            pair1 = [pair[0], pair[1]]
            pair2 = [pair[1], pair[0]]
            same_as_graph[idx % self._batch_size].append(pair1)
            same_as_graph[idx % self._batch_size].append(pair2)


        for batch_idx, batch_pair_nodes in enumerate(same_as_graph):

            nodes_to_select = torch.from_numpy(np.array(batch_pair_nodes)).flatten().unique()
            indexes_to_retrive_family = torch.isin(families, nodes_to_select).any(dim=0)
            final_nodes = families[:, indexes_to_retrive_family].flatten().unique()
            batch_subgraph, negative_batch_subgraph = self.extract_subgraph(population=final_nodes, edges_type=possible_edges)
            
            images_to_keep = torch.from_numpy(self._graph["lines"][final_nodes])

            yield batch_subgraph, negative_batch_subgraph, images_to_keep, final_nodes





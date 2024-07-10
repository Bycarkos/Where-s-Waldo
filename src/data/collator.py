## Custom Imports
from data.dataset import Graphset
from data.volumes import *

import torch
from  torch_geometric.sampler import HeteroSamplerOutput
from torch_geometric.sampler import BaseSampler
from torch_geometric.data import HeteroData


from typing import *


import pickle
import copy
import random 
import json
import glob
import numpy as np



class FamilyCollator():
    
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

    def __init__(self, graph: Type[Graphset], batch_size:int=64, partition_ration: Tuple[float, ...] = [0.8, 0.1, 0.1], shuffle:bool=True) -> None:
        
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
        self._graph_batch = [[] for i in range(batch_size)]
        self._same_as_graph = copy.copy(self._graph_batch)

        #extract type of nodes


        self.__create_partition()


    def __create_partition(self):
        
        """
        Creates partitions for training, validation, and test sets based on 'same as' edges.
        """   
        
        same_as_edge_index = self._graph[("individuals", "same_as", "individuals")].edge_index

        train, validation, test = [], [], []
        
        index_permutation = torch.randperm(same_as_edge_index.shape[1])
        train_partition  = index_permutation[:int(index_permutation.shape[0]*self._train_ratio)]
        validation_partition = index_permutation[int(index_permutation.shape[0]*self._train_ratio): int(index_permutation.shape[0]*(self._train_ratio+ self._validation_ratio))]
        test_partition = index_permutation[int(index_permutation.shape[0]*(self._train_ratio+ self._validation_ratio)):]


        ## Edge index partitions with Same As
        self._same_as_train = same_as_edge_index[:, train_partition]
        self._same_as_validation = same_as_edge_index[:, validation_partition]
        self._same_as_test = same_as_edge_index[:, test_partition]

        ## Nodes partitions from same as
        self._ind_nodes_train = self._same_as_train.flatten()
        self._ind_nodes_validation = self._same_as_validation.flatten()
        self._ind_nodes_test = self._same_as_test.flatten()





    def collate_test(self):
    
        """
        Collates the test set subgraph.

        Returns:
            Tuple[Dict, torch.Tensor]: A tuple containing the subgraph and the final nodes.
        """
        
        families = self._graph[("individuals", "family", "individuals")].edge_index
        indexes_to_retrive_family = torch.isin(families, self._same_as_test).any(dim=0)
        final_nodes = families[:, indexes_to_retrive_family].flatten().unique()
        
        
        possible_edges = (self._graph.edge_index_dict.keys())
        
        subgraph = self.extract_subgraph(population=final_nodes, possible_edges=possible_edges)
        
        return subgraph, final_nodes
    
    def collate_validation(self):
        
        """
        Collates the validation set subgraph.

        Returns:
            Tuple[Dict, torch.Tensor]: A tuple containing the subgraph and the final nodes.
        """
        
        families = self._graph[("individuals", "family", "individuals")].edge_index
        indexes_to_retrive_family = torch.isin(families, self._ind_nodes_validation).any(dim=0)
        final_nodes = families[:, indexes_to_retrive_family].flatten().unique()

        
        
        possible_edges = (self._graph.edge_index_dict.keys())
        
        subgraph = self.extract_subgraph(population=final_nodes, possible_edges=possible_edges)
        
        return subgraph, final_nodes

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
            print(possible_edge)
                
            adj = self._graph[possible_edge].edge_index
            negative_sampling = self._graph[possible_edge].negative_sampling

            indexes_to_retrieve = torch.isin(adj, population).all(dim=0)
            filtered_adj = adj[:, indexes_to_retrieve]
            
            if len(negative_sampling) != 0:
                negative_indexes_to_retrive = torch.isin(negative_sampling, population).all(dim=0)
                negative_subgraph[possible_edge] = negative_sampling[:, negative_indexes_to_retrive]

            subgraph[possible_edge] = filtered_adj
            
        
        return subgraph, negative_subgraph
            
            
        

    def __iter__(self):
        
        
        """
        Iterator to generate batches of subgraphs for training.

        Yields:
            Tuple[Dict, torch.Tensor]: A tuple containing a batch subgraph and the final nodes for each batch.
        """
        
        distribution = [ pair for pair in self._same_as_train.numpy().T]
        random.shuffle(distribution)


        families = self._graph[("individuals", "family", "individuals")].edge_index
        possible_edges = (self._graph.edge_index_dict.keys())
        

        for idx, pair in enumerate(distribution):
            self._same_as_graph[idx % self._batch_size].append(pair)
            self._graph_batch[idx % self._batch_size].append(pair)


        for batch_idx, batch_pair_nodes in enumerate(self._same_as_graph):
            nodes_to_select = torch.from_numpy(np.array(batch_pair_nodes)).flatten().unique()
            indexes_to_retrive_family = torch.isin(families, nodes_to_select).any(dim=0)
            final_nodes = families[:, indexes_to_retrive_family].flatten().unique()
            batch_subgraph, negative_batch_subgraph = self.extract_subgraph(population=final_nodes, edges_type=possible_edges)
            
            
            yield batch_subgraph, negative_batch_subgraph, final_nodes




if __name__ == "__main__":

    volume_1889_1906 = [Path("data/CED/SFLL/1889"), Path("data/CED/SFLL/1906")]

    entities = ["nom", "cognom_1", "cognom_2", "parentesc", "ocupacio"]

    ## volumes structure
    volums =  []

    ## Pages Structure
    for auxiliar_volume in volume_1889_1906:  
        print("STARTING DOWNLOADING VOLUMES: VOLUME-", auxiliar_volume)
        pages_path = sorted([f.path for f in os.scandir(auxiliar_volume) if f.is_dir()])
        with open(Path(auxiliar_volume, "graph_gt_corroborator.json")) as file:
            graph_gt = json.load(file)

        count= 0
        pages = []
        for idx, page_folder in enumerate(pages_path):
            gt_page = pd.read_csv(page_folder + "/gt_alignement.csv")


            ##extract the information of the lines first
            # condition to remove pages with inconsitencis
            page_lines = []
            page = Image.open(page_folder+".jpg")

            n_families = graph_gt[os.path.basename(page_folder)+".jpg"]["families"]

            if len(glob.glob(os.path.join(page_folder, "row_*"))) != graph_gt[os.path.basename(page_folder)+".jpg"]["individus"]:
                count += 1
                continue
            
            else:
                lines_page = (glob.glob(os.path.join(page_folder, "row_*")))
                sorted_lines = sorted_file_names = sorted(lines_page, key=sort_key)
                with open(page_folder + "/info.json", "rb") as file:
                    bboxes = json.load(file)["rows_bbox"]

                for idx_line, path_line in enumerate(sorted_lines):
                    ocr = gt_page.loc[idx_line, entities].values
                    bbox_line = bboxes[idx_line]
                    page_lines.append(Line(Path(path_line), bbox_line, bbox_line, ocr))



            pages.append(Page(Path(page_folder+".jpg"), [0, 0, *page.size ], page_lines, n_families))
            
        volums.append(Volume(auxiliar_volume, pages, entities))


    pk = {"Noms_harmo":"nom", "cognom1_harmo":"cognom_1", "cognom2_harmo":"cognom_2", "parentesc_har":"parentesc", "ocupacio":"ocupacio"}

    Graph = Graphset(Volumes=volums,auxiliar_entities_pk=pk)


    print(Graph._graph)
    test = Graph._graph[("cognom_2", "similar", "cognom_2")].edge_index
    #print(Graph._total_gt.iloc[test.flatten().unique().numpy()][["nom", "cognom_1", "cognom_2", "parentesc", "ocupacio"]])
    collator = FamilyCollator(graph=Graph)

    ##### TODO Acabar de fer el test per a que funcioni el collator.
    #print(graph._graph)
    for i in collator:
        #edge_ind = i[("cognom_2", "similar", "cognom_2")].edge_index
        print(i)
        print(i[("cognom_1", "similar", "cognom_1")].shape)
        exit()

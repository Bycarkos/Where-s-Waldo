
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

## commmon packages
import pandas as pd
import os
import glob
from PIL import Image
import itertools
import re


    

class Graphset(Dataset):
    
    """
    Graphset is a dataset class responsible for creating and initializing a heterogeneous graph 
    from multiple volumes of data. This class manages the core information, constructs the graph,
    and initializes various types of nodes and edges.

    Attributes:
        _volumes (List[Volume]): List of Volume objects containing the data.
        _total_attributes_nodes (int): Total number of attribute nodes.
        _total_individual_nodes (int): Total number of individual nodes.
        _total_entities (set): Set of unique entities across all volumes.
        _total_gt (DataFrame): Merged ground truth data from all volumes.
        _graph (HeteroData): Heterogeneous graph data structure.

    Methods:
        __initialize_core_information(): Initializes core information from the volumes.
        __initialize_nodes(): Initializes nodes in the graph.
        __initialize_individual_attribute_edges(auxiliar_search_pk): Initializes edges between individuals and attributes.
        __initialize_attribute_edges(auxiliar_search_pk): Initializes attribute edges based on harmonized data.
        __initialize_same_as_edges(): Initializes 'same as' edges between individuals.
        __intitlialize_family_edges(): Initializes family edges between individuals.
        __load_images_on_graph(): Loads images into the graph structure.
        generate_families(index_list): Generates family edges from a list of indexes.
        generate_index_permutations(index_list): Generates permutations of indexes for creating edges.
        contains_letters_and_numbers(s): Checks if a string contains both letters and numbers.
    """

    def __init__(self, Volumes: List[Volume], auxiliar_entities_pk: Optional[Tuple[str, ...]] = None, add_edge_features:bool= True) -> None:
        
        """
        Initializes the Graphset with volumes of data and optional auxiliary entities.

        Args:
            Volumes (List[Volume]): List of Volume objects containing the data.
            auxiliar_entities_pk (Optional[Tuple[str, ...]]): Optional tuple of auxiliary entity primary keys.
        """
        
        
        super(Graphset, self).__init__()

        self._volumes = Volumes
        
        #initialize core information such as the total of attributes, individuals, entites etc...
        self.__initialize_core_information()
        self.__mapping_pk = auxiliar_entities_pk 
        self._add_edge_features = add_edge_features
        print("Lines good Recovered with no permutation inconsitences: ", self._total_individual_nodes.item())
        print("Total Lines : ", sum(len(volume.gt()) for volume in self._volumes))
        print("Total proportion : ", (self._total_individual_nodes/sum(len(volume.gt()) for volume in self._volumes)).item(), "%")


        ### Core Graph construction 
        self._graph = HeteroData()

    
    
    def __initialize_core_information(self):
        
        """
        Initializes core information from the volumes, including attribute nodes, individual nodes, and entities.
        """
        
        
        self._total_attributes_nodes = 0
        self._total_individual_nodes = 0
        self._total_entities = set([])

        for volume in self._volumes:
            self._total_attributes_nodes += torch.tensor(volume.count_attributes())
            self._total_individual_nodes += torch.tensor(volume.count_individuals())
            self._total_entities = self._total_entities.union(set(volume._entities))


        ## merge the grountruth of the different volumes
        self._total_gt = pd.concat([page.gt() for volume in self._volumes for page in volume._pages], axis=0)


    

    def _initialize_nodes(self, embedding_size:int=128):
        
        """
        Initializes nodes in the graph with placeholders for each kind of node attribute and all attributes.
        """
        ## placeholding for each kind of node attribute
        for etype in self._total_entities:
            self._graph[etype].x = torch.zeros((self._total_individual_nodes, embedding_size), dtype=torch.float32)

        ## ADD the image transformation placeholder falta carregar les imatges (hauria de carregar les imatges en el __get__item)
        ## placeholding for all the lines(indivduals)
        self._graph["individuals"].x = torch.zeros((self._total_individual_nodes, embedding_size), dtype=torch.float32)

    def _initialize_image_information(self):
        """
        Loads images into the graph structure.
        """
        
        image_lines_path = []
        image_lines = []
        widths = []
        heights = []
        ar = []

        for volume in self._volumes:
            for page in volume._pages:
                for line in page._individuals:

                    image_lines_path.append(line) 
                    image = T.to_tensor(line.image())
                    heights.append(image.shape[1])
                    widths.append(image.shape[2])
                    ar.append(image.shape[2] / image.shape[1])

                    image_lines.append(line)
                    
        self._graph["lines"] = np.array(image_lines_path)
        self._graph["image_lines"] = image_lines
        self._graph["max_width"] = torch.tensor(max(widths))
        self._graph["min_width"] = torch.tensor(min(widths))

        self._graph["max_height"] = torch.tensor(max(heights))
        self._graph["min_height"] = torch.tensor(min(heights))

        self._graph["min_aspect_ratio"] = torch.tensor(min(ar))
        self._graph["max_aspect_ratio"] = torch.tensor(max(ar))

        self._graph["mean_width"] = torch.tensor(np.mean(widths))
        self._graph["std_width"] = torch.tensor(np.std(widths))

        self._graph["mean_height"] = torch.tensor(np.mean(heights))
        self._graph["std_height"] = torch.tensor(np.std(heights))


    def _initialize_edges(self, embedding_size:int=128):
        self.__initialize_individual_attribute_edges(embedding_size=embedding_size) ## every line has the sma index name and surname because is a replication
        self.__initialize_attribute_edges(auxiliar_search_pk=self.__mapping_pk)
        self.__initialize_same_as_edges()
        self.__intitlialize_family_edges()



    def __initialize_individual_attribute_edges(self, embedding_size:int=128):
        
        """
        Initializes edges between individuals and attributes.

        Args:
            auxiliar_search_pk (Optional[Tuple[str, ...]]): Optional tuple of auxiliary entity primary keys.
        """

        ## placeholding
        total_entities_pk = (self._total_entities)

        for idx, etype in (enumerate(total_entities_pk)):
            edge_key = ("individuals", etype, etype)
            edge_index = list(zip(range(self._total_individual_nodes), range(self._total_individual_nodes)))
            edge_index = torch.tensor(edge_index).T

            self._graph[edge_key].edge_index = edge_index
            self._graph[edge_key].negative_sampling = []

            if self._add_edge_features is True:  
                self._graph[edge_key].edge_attr =  torch.zeros((embedding_size), dtype=torch.float32)
   

    def __initialize_attribute_edges(self, auxiliar_search_pk:Optional[Dict[str, str]] = None):
        
        """
        Initializes attribute edges based on harmonized data.

        Args:
            auxiliar_search_pk (Optional[Dict[str, str]]): Optional dictionary of auxiliary search primary keys.
        """
        
        ## placeholding

        total_entities_pk = (self._total_entities) if auxiliar_search_pk == None else auxiliar_search_pk
        total_nodes = np.arange(self._total_individual_nodes)

        for idx, (harmo, etype) in (enumerate(total_entities_pk.items())):
            
            ### Group by the harmonized data
            grouped = self._total_gt.groupby(harmo, as_index=False)
            edge_index = []
            ns_edge_index =  []


            for group_name, _ in  grouped:
                index_group_by = grouped._get_index(group_name)
                negative_sampling_nodes_idx = np.isin(total_nodes, index_group_by)
                negative_sampling_nodes_idx = negative_sampling_nodes_idx != True
                negative_sampling = np.array(list(itertools.product(index_group_by, total_nodes[negative_sampling_nodes_idx])))

                edge_index.extend(self.generate_index_permutations(grouped._get_index(group_name)))
                ns_edge_index.extend(negative_sampling)
                
            edge_index = torch.tensor(edge_index).T
            ns_edge_index = np.array(ns_edge_index)
            ns_edge_index = torch.from_numpy(ns_edge_index).T
            
            ## select the non harmonized data
            self._graph[(etype, "similar", etype)].edge_index = edge_index
            self._graph[(etype, "similar", etype)].negative_sampling = ns_edge_index
            ## Extract all possible negative sampling for each individual
            
    def __initialize_same_as_edges(self):
        
        """
        Initializes 'same as' edges between individuals based on ground truth data.
        """

        true_pairs = []

        ## Filter as possible
        self._total_gt["id"] = self._total_gt["id"].fillna("SFLL")

        df = self._total_gt[self._total_gt['id'].apply(self.contains_letters_and_numbers)]
        grouped = df.groupby('id', as_index=False)

        for group_name, group in  grouped:
            true_pairs.extend(self.generate_index_permutations(grouped._get_index(group_name)))

        true_pairs = torch.tensor(true_pairs).T
        self._graph[("individuals", "same_as", "individuals")].edge_index = true_pairs
        self._graph[("individuals", "same_as", "individuals")].negative_sampling = self._graph[("cognom_2", "similar", "cognom_2")].edge_index


    def __intitlialize_family_edges(self):
        
        """
        Initializes family edges between individuals based on ground truth data.
        """

        families = []

        ## Filter as possible

        df = self._total_gt[self._total_gt['id_llar_general'].apply(self.contains_letters_and_numbers)]
        grouped = df.groupby('id_llar_general', as_index=False)

        for group_name, group in  grouped:
            real_indexes = grouped._get_index(group_name)
            parentesc = (group["parentesc"].values)
            families.extend(self.generate_families(list(zip(parentesc, real_indexes))))
        
        families = torch.tensor(families).T
        self._graph[("individuals", "family", "individuals")].edge_index = families
        self._graph[("individuals", "family", "individuals")].y = torch.ones(families.shape[1])
        self._graph[("individuals", "family", "individuals")].negative_sampling = []




    def generate_families(self, index_list):
        
        """
        Generates family edges from a list of indexes.

        Args:
            index_list (List[Tuple[str, int]]): List of tuples containing family roles and indexes.

        Returns:
            List[Tuple[int, int]]: List of family edges.
        """
        
        if len(index_list) > 1:
            _, real_indexes = list(zip(*index_list))
            real_indexes = list(real_indexes)
            dic_families = dict(index_list)

            index_jefe = dic_families.get("jefe", None)
            if index_jefe is not None:
                real_indexes.remove(index_jefe)

                return [(index_jefe, idx_pair) for idx_pair in real_indexes]
            else:
                return self.generate_index_permutations(real_indexes)

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



if __name__ == "__main__":
    import json
    import glob

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






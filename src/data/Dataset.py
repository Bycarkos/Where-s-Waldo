
## Custom packages
from Volumes import Volume, Page, Line, sort_key


## geometrical data packages
import dgl
import torch
import torchvision
from torch_geometric.data import HeteroData
from torch.utils.data import Dataset, DataLoader

## Typing packages
from torchtyping import TensorType, patch_typeguard
from typing import *
from pathlib import Path

## commmon packages
import pandas as pd
import os
import glob
from PIL import Image
from tqdm import tqdm
import itertools
import re



    

class Graphset(Dataset):

    def __init__(self, Volumes: List[Volume], auxiliar_entities_pk: Optional[Tuple[str, ...]] = None) -> None:
        super(Graphset, self).__init__()

        self._volumes = Volumes
        
        #initialize core information such as the total of attributes, individuals, entites etc...
        self.__initialize_core_information()

        print("Lines good Recovered with no permutation inconsitences: ", self._total_individual_nodes.item())
        print("Total Lines : ", len(self._volumes[-1].gt()))
        print("Total proportion : ", (self._total_individual_nodes/len(self._volumes[-1].gt())).item(), "%")



        ### Core Graph construction 
        self._graph = HeteroData()

        self.__initialize_nodes()
        self.__initialize_attribute_edges(auxiliar_search_pk=auxiliar_entities_pk)
        exit()
        self.__intitlialize_family_edges()
        exit()
        self.__initialize_same_as_edges()
        exit()
        self.__initialize_attribute_edges(auxiliar_search_pk=auxiliar_entities_pk)

    
    
    def __initialize_core_information(self):
        self._total_attributes_nodes = 0
        self._total_individual_nodes = 0
        self._total_entities = set([])

        for volume in self._volumes:
            self._total_attributes_nodes += torch.tensor(volume.count_attributes())
            self._total_individual_nodes += torch.tensor(volume.count_individuals())
            self._total_entities = self._total_entities.union(set(volume._entities))

        ## merge the grountruth of the different volumes
        self._total_gt = pd.concat([page.gt() for volume in self._volumes for page in volume._pages], axis=0)

    def __initialize_nodes(self):
        
        ## placeholding for each kind of node attribute
        for etype in self._total_entities:
            self._graph[etype].x = torch.zeros(self._total_individual_nodes, dtype=torch.int8)
        
        ## placeholding for all the attributes (concatenate the previous embeddings)
        self._graph["attributes"].x = torch.zeros(self._total_attributes_nodes, dtype=torch.int8)    



        ## ADD the image transformation placeholder
        ## placeholding for all the lines(indivduals)
        self._graph["individuals"].x = torch.zeros(self._total_individual_nodes, dtype=torch.int8)


    def __initialize_attribute_edges(self, auxiliar_search_pk:Optional[Tuple[str, ...]] = None):
        ## placeholding
        n = self._total_individual_nodes

        total_entities_pk = (self._total_entities) if auxiliar_search_pk == None else auxiliar_search_pk

        for idx, etype in (enumerate(total_entities_pk)):

            edge_index = []
            print(self._total_gt[etype].head(6))
            grouped = self._total_gt.groupby(etype, as_index=False)
            for group_name, group in  grouped:
                edge_index.extend(self.generate_index_permutations(grouped._get_index(group_name)))
            
            edge_index = torch.tensor(edge_index).T

            self._graph[(etype, "similar", etype)].edge_index = edge_index


    # TODO acabar de revisar
    def __initialize_same_as_edges(self):

        true_pairs = []

        ## Filter as possible
        self._total_gt["id"] = self._total_gt["id"].fillna("SFLL")

        df = self._total_gt[self._total_gt['id'].apply(self.contains_letters_and_numbers)]
        grouped = df.groupby('id', as_index=False)

        for group_name, group in  grouped:
            true_pairs.extend(self.generate_index_permutations(grouped._get_index(group_name)))
        exit()
        true_pairs = torch.tensor(true_pairs).T
        print(true_pairs)
        exit()
        self._graph[("individuals", "same_as", "individuals")].edge_index = true_pairs



    def __intitlialize_family_edges(self):

        families = []

        ## Filter as possible

        df = self._total_gt[self._total_gt['id_llar_general'].apply(self.contains_letters_and_numbers)]
        grouped = df.groupby('id_llar_general', as_index=False)

        for group_name, group in  grouped:
            families.extend(self.generate_index_permutations(grouped._get_index(group_name)))
        
        families = torch.tensor(families).T
        print(families)
        exit()
        self._graph[("individuals", "family", "individuals")].edge_index = families


    @staticmethod
    def generate_index_permutations(index_list):
        if len(index_list) > 1:
            return [pair for pair in itertools.permutations(index_list, 2) if pair[0] != pair[1]]
        else:
            return []

    @staticmethod
    def contains_letters_and_numbers(s):
        print(s)
        return bool(re.search(r'[a-zA-Z]', s)) and bool(re.search(r'[0-9]', s))



if __name__ == "__main__":
    import json
    import glob

    volume_1889 = Path("data/CED/SFLL/1889")

    entities = ["nom", "cognom_1", "cognom_2", "parentesc", "ocupacio"]

    ## volumes structure
    volums =  []

    ## Pages Structure
    
    pages_path = sorted([f.path for f in os.scandir(volume_1889) if f.is_dir()])
    with open(Path(volume_1889, "graph_gt_corroborator.json")) as file:
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
        
    volums.append(Volume(volume_1889, pages, entities))


    pk = ["Noms_harmo", "cognom1_harmo", "cognom2_harmo", "parentesc_har", "ocupacio"]

    Graph = Graphset(Volumes=volums,auxiliar_entities_pk=pk)





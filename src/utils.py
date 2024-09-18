import torch
import torch.nn as nn

from  torchtyping import TensorType

import os
import umap
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st 
import fasttext
import pdb

import torch_geometric.utils as tutils
from torch_geometric.data import HeteroData

from typing import *

import tqdm
import pickle
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import fasttext.util

device = "cuda" if torch.cuda.is_available() else "cpu"


def write_pickle(info: Any, filepath: str) -> None:
    os.makedirs(os.path.split(filepath)[0], exist_ok=True)
    
    with open(filepath, "wb") as file:
        pickle.dump(info, file)
        
def read_pickle(filepath:str) -> Any:
    
    with open(filepath, "rb") as file:
        obj = pickle.load(file=file)
        
    return obj
        


def extract_subgraph(graph:Type[HeteroData], 
                    population:List[Tuple[int, int]],
                    edges_to_extract: List[Tuple[str, ...]]) -> Dict:
    
    """
    Extracts a subgraph based on the specified population of nodes and edge types.

    Args:
        population (List[Tuple[int, int]]): List of nodes to include in the subgraph.
        edges_type (List[Tuple[str, str, str]]): List of edge types to include in the subgraph.

    Returns:
        Dict: A dictionary containing the filtered adjacency lists for each edge type.
    """
    
    subgraph = {}
    
    edge_types = graph.edge_index_dict.keys()
    for possible_edge in edges_to_extract:
            
        adj = graph[possible_edge].edge_index.to(device)
        specific_subgraph, nodes = tutils.subgraph(subset=population.type(torch.int64), edge_index=adj.type(torch.int64))
        
        #if len(negative_sampling) != 0:
        #    negative_indexes_to_retrive = torch.isin(negative_sampling, population).all(dim=0)
        #    negative_subgraph[possible_edge] = negative_sampling[:, negative_indexes_to_retrive]

        subgraph[possible_edge] = specific_subgraph
        
    
    return subgraph



def calc_euclidean(x1, x2):
    return (x1 - x2).pow(2).sum(1)


def contrastive_loss(x1, x2, label, margin: float = 1.0):
    """
    Computes Contrastive Loss
    """

    dist = torch.nn.functional.pairwise_distance(x1, x2)

    loss = (1 - label) * torch.pow(dist, 2) \
        + (label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    loss = torch.mean(loss)

    return loss


def compute_resize(shape, patch_size):
    offset = shape % patch_size
    if offset > (patch_size /2):
        return shape + (patch_size - offset)
    else:
        return shape - offset
    
    
def extract_optimal_shape(min_width:int,  max_height:int, patch_size:int):
    
    min_width_recovered = compute_resize(shape=min_width, patch_size=patch_size)
    max_height_recovered = compute_resize(shape=max_height, patch_size=patch_size)
    
    return int(min_width_recovered), int(max_height_recovered)
    

def compute_confidence_interval(data:list, alpha:float):
    mean = np.mean(data)
    scale = st.sem(data)
    return st.t.interval(confidence=alpha, 
                df=len(data)-1, 
                loc=mean,  
                scale=scale)





@torch.no_grad()
def extract_embeddings_from_graph(loader ,
                                    graph,
                                    model):
    
    model.eval()
    attribute_embeddings = graph.x_attributes.to(device)
    entity_embeddings = graph.x_entity.to(device)
    
    for idx, dict_images in tqdm.tqdm(enumerate(loader), desc="Extracting the embeddings from the model"):
        images = dict_images["image_lines"].to(device)
        ocr_indexes = dict_images["ocrs"].to(device)
        population = dict_images["population"].to(device)

        attribute_representation, individual_embeddings = model(x=images)#.encode_attribute_information(image_features=image_features, edge_attributes=edge_features) #message_passing
        attribute_embeddings[population] = attribute_representation
        if individual_embeddings is not None:
            entity_embeddings[population, 0] = individual_embeddings

    return attribute_embeddings, entity_embeddings

def extract_language_embeddings(list_possible_ocr: list, embedding_size: int, filepath:str):
    
    ft = fasttext.load_model('cc.ca.300.bin')
    fasttext.util.reduce_model(ft, embedding_size)
    embedding_to_save = []
    
    for idx in tqdm.tqdm(range(len(list_possible_ocr)), desc="Extracting Language Embeddings"):
        ocr = list_possible_ocr[idx]
        try:
            word_embedding = ft.get_word_vector(ocr)
        except:
            word_embedding = ft.get_word_vector("no consta")

        embedding_to_save.append(word_embedding)
            
            
    write_pickle(info=embedding_to_save, filepath=filepath)
    
    return embedding_to_save

@torch.no_grad()
def evaluate_attribute_metric_space(attribute_embeddings: np.ndarray,
                                    plot_path:str = 'embedding_metric_space.png'):

    
    gt = np.zeros((attribute_embeddings.shape[0], attribute_embeddings.shape[1]))
    final_embeddings = attribute_embeddings.reshape(-1, attribute_embeddings.shape[-1])

    

    for i in range(attribute_embeddings.shape[1]):
        gt[:, i] += i

    gt = gt.flatten()

    kf = KFold(n_splits=10)

        
    # Define the different values of n_neighbors
    n_neighbors_list = [1, 3, 5, 10]

    metrics = {
        "inter_att_dist": {}
    }
    # Iterate over the list of n_neighbors and evaluate the model

    acc, f_score = {k: [] for k in n_neighbors_list}, {k: [] for k in n_neighbors_list}

    for i, (train_index, test_index) in enumerate(kf.split(final_embeddings, gt)):
        print(f"Fold {i}:")

        x_embeddings_train = final_embeddings[train_index]
        x_embeddings_test = final_embeddings[test_index]

        y_train = gt[train_index]
        y_test = gt[test_index]

        for n_neighbors in n_neighbors_list:
            print(f"\nSTARTING THE EVALUATION WITH KNN, n_neighbors = {n_neighbors}")
            neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
            neigh.fit(x_embeddings_train, y_train)
        
            predictions = neigh.predict(x_embeddings_test)
        
            # Calculate the metrics
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='macro')

            acc[n_neighbors].append(accuracy)
            f_score[n_neighbors].append(f1)

        metrics["inter_att_dist"] = {"Accuracy_dist": {k: (np.mean(ac), np.std(ac)) for k, ac in acc.items()},
                                    "F_score_dist":  {k: (np.mean(f), np.std(f)) for k, f in f_score.items()}}
    return metrics
    
    
    
@torch.no_grad()
def extract_intra_cluster_nearest_neighboors_at_k(x: TensorType,
                                                  top_k=10):
    #Graph = utils.read_pickle("pickles/graphset_3_volumes_128_entities_4.pkl")

    dict_nearest_neighbors = {}
    distances_dict = {}
    for idx in range(x.shape[1]):
        embeddings = x[:, idx]
        distances = torch.cdist(embeddings, embeddings, p=2)
        distances_dict[idx] = distances
        
        # Set the diagonal to a large positive number to avoid self-matching
        distances.fill_diagonal_(float('inf'))
        # Get the top K nearest neighbors for each embedding (smallest distances)
        top_k_values, top_k_indices = torch.topk(distances, top_k, dim=1, largest=False)

        # Create a dictionary mapping each index to its top K nearest neighbors
        nearest_neighbors = {ix: top_k_indices[ix].tolist() for ix in range(distances.size(0))}

        dict_nearest_neighbors[idx] = nearest_neighbors
    
    return dict_nearest_neighbors, distances_dict              


import models.attentions as att



import torch.nn as nn
import torch
from torch_geometric.nn import HGTConv, HeteroConv, GCNConv, SAGEConv, GATConv, Linear

from torch_geometric.typing import Adj
from torchtyping import TensorType
from typing import *

import pdb


from omegaconf import DictConfig, OmegaConf


device = "cuda" if torch.cuda.is_available() else "cpu"



class ForwardAttributeGnn(nn.Module):
    
    def __init__(self, embedding_size:int, n_different_edges:int):
          
        super().__init__()
        

        self.n_different_edges = n_different_edges
        self.embedding_size = embedding_size
        self._activation = nn.ReLU(inplace=False)

        self._edge_space_projection = nn.Parameter(data=torch.randn(size=(n_different_edges, self.embedding_size, self.embedding_size)), requires_grad=True).to(device=device)
        nn.init.kaiming_uniform_(self._edge_space_projection)

        self._edge_space_aggregator = nn.Linear(in_features=self.embedding_size, out_features=self.embedding_size)

    
    def compute_edge_message(self, src_embeddings: TensorType["batch", "|E|", "d"]):
        
        aggregated_information : TensorType["batch", "|E|", "d"] = self._edge_space_aggregator(src_embeddings) #(B, self._attributes, embeddingsize )
        edge_space_projector : TensorType["batch", "|E|", "d"] = torch.einsum('bai,aij->baj', aggregated_information, self._edge_space_projection)

        return edge_space_projector

    def forward(self, x:TensorType["batch", "|E|", "d"]):


        forward_message = self.compute_edge_message(src_embeddings=x)
        

        return forward_message     
    
    

class BackwardAttributeGnn(nn.Module):
    
    
    def __init__(self, embedding_size:int, n_different_edges:int):
          
        super().__init__()

        self.embedding_size = embedding_size
        self.n_different_edges = n_different_edges
        self._activation = nn.ReLU(inplace=False)
        self._edge_space_projection = nn.Parameter(data=torch.randn(size=(n_different_edges, self.embedding_size, self.embedding_size)), requires_grad=True).to(device=device)
        self._self_weighted = nn.Parameter(data=torch.ones(size=(1, n_different_edges, 1)), requires_grad=True).to(device=device)
        
        nn.init.kaiming_uniform_(self._edge_space_projection)

        self._Q = nn.Linear(in_features=embedding_size, out_features=embedding_size)
        self._K = nn.Linear(in_features=embedding_size, out_features=embedding_size)
        self._V = nn.Linear(in_features=embedding_size, out_features=embedding_size)

        self._attention_mechanism = att.ScaledDotProductAttention()


    def compute_semantic_message(self, attribute_embeddings: TensorType["batch", "attribute_nodes", "embedding_size"]):
        
        
        #[b a, i]
        #[a, i, j ]
        edge_space_projector = torch.einsum('bai,aij->baj', attribute_embeddings, self._edge_space_projection)
        edge_space_projector = self._activation(edge_space_projector)
        
        queries = self._Q(edge_space_projector)
        keys = self._K(edge_space_projector)
        values = self._V(edge_space_projector)
        
        # TensorType["B", "d"]
        contextual_entity_embedding, self._attentions = self._attention_mechanism(query=queries,
                                                                key=keys,
                                                                value=values)
        contextual_entity_embedding: TensorType["B", "d"] = torch.sum(contextual_entity_embedding, dim=1)

        return contextual_entity_embedding
    
        
    def forward(self, attribute_embeddings: TensorType["B", "|E|", "d"]):
        
        embeddings_individuals = self.compute_semantic_message(attribute_embeddings=attribute_embeddings)

        return embeddings_individuals  
    

    
class AttributeGNN(nn.Module):
    
    def __init__(self, embedding_size:int, n_different_edges:int) -> None:
        super().__init__()
        
        self._forward_attribute_message = ForwardAttributeGnn(embedding_size=embedding_size, n_different_edges=n_different_edges)
        self._backward_attribute_message = BackwardAttributeGnn(embedding_size=embedding_size, n_different_edges=n_different_edges)
        
    
    def forward(self, 
                x: TensorType["batch", "|E|", "d"]) -> Tuple[TensorType["batch", "|E|", "d"],
                                                              TensorType["batch", "d"]]: # type: ignore
        

        attribute_representation = self._forward_attribute_message(x)

        individual_embeddings = self._backward_attribute_message(attribute_representation)

        return attribute_representation, individual_embeddings
        
    
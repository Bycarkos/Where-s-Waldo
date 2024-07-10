
import torch.nn as nn
import torch

from torch_geometric.nn import HGTConv, HeteroConv, GCNConv, SAGEConv, GATConv, Linear

from torch_geometric.typing import Adj
from torchtyping import TensorType
from typing import *



class MMGCM(nn.Module):

    def __init__(self, visual_encoder: Type[nn.Module], gnn_encoder: Type[nn.Module], edge_encoder: Optional[Type[nn.Module]] = None) -> None:
        super(MMGCM).__init__()


        self._visual_encoder = visual_encoder
        self._gnn_encoder = gnn_encoder
        self._edge_encoder = edge_encoder


    def encode_visual_information(self, x: TensorType["batch", "(C, H, W)"]):
        return self._visual_encoder(x)
    
    
    def encode_edge_positional_information(self, x: TensorType["batch", "(C, H, W)"]):
        return self._edge_encoder(x)
    
    def update_embeddings_with_message_passing(self, x: Dict[str, TensorType],
                edge_index:Dict[Tuple[str, str, str], Adj], 
                population: TensorType["batch_individual_nodes"],
                edge_attributes: Optional[TensorType["batch", "Nattributes", "embedding_size"]]):
        
        return self._gnn_encoder(x_dict=x, edge_index_dict=edge_index, edge_attributes=edge_attributes, population=population)
    
            
    def forward(self):
        pass
        

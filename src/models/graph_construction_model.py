
import torch.nn as nn
import torch

from torch_geometric.nn import HGTConv, HeteroConv, GCNConv, SAGEConv, GATConv, Linear

from torch_geometric.typing import Adj
from torchtyping import TensorType
from typing import *

from omegaconf import DictConfig


class MMGCM(nn.Module):

    def __init__(self, cfg:DictConfig, visual_encoder: Type[nn.Module], gnn_encoder: Type[nn.Module], edge_encoder: Optional[Type[nn.Module]] = None) -> None:
        super(MMGCM, self).__init__()


        self._visual_encoder = visual_encoder(cfg=cfg.visual_encoder)
        self._gnn_encoder = gnn_encoder(cfg=cfg.gnn_encoder)
        self._edge_encoder = edge_encoder(cfg = cfg.edge_visual_encoder)


    def encode_visual_information(self, x: TensorType["batch", "(C, H, W)"]):
        return self._visual_encoder(x)
    
    
    def encode_edge_positional_information(self, x: TensorType["batch", "(C, H, W)"]):
        return self._edge_encoder(x)
    
    
    def encode_attribute_information(self,image_features:[TensorType["batch", "embedding_size"]],
                edge_attributes: [TensorType["batch", "Nattributes", "embedding_size"]]):        
        return self._gnn_encoder(image_features=image_features, edge_attributes=edge_attributes)
    


            
    def forward(self, x: TensorType["batch", "(C, H, W)"]):

        image_features = self.encode_visual_information(x)
        edge_features = self.encode_edge_positional_information(x)

        attribute_representation, individual_embeddings = self.encode_attribute_information(image_features, edge_features)

        return attribute_representation, individual_embeddings
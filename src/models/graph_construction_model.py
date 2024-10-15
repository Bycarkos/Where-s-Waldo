
import torch.nn as nn
import torch

from torch_geometric.nn import HGTConv, HeteroConv, GCNConv, SAGEConv, GATConv, Linear

from torch_geometric.typing import Adj
from torchtyping import TensorType
from typing import *

from omegaconf import DictConfig


class MMGCM(nn.Module):

    def __init__(self, visual_encoder: Type[nn.Module], gnn_encoder: Type[nn.Module],
                  edge_encoder: Optional[List[Type[nn.Module]]] = None) -> None:
        super(MMGCM, self).__init__()


        self._visual_encoder = visual_encoder
        self._gnn_encoder = gnn_encoder

        self._edge_encoders = edge_encoder


    def load_weights_visual_encoder(self, path):
        self._visual_encoder.load_state_dict(torch.load(path))
    
    def load_weights_edge_visual_encoder(self, path):
        self._edge_encoder.load_state_dict(torch.load(path))
    
    def load_weights_gnn_encoder(self, path):
        self._gnn_encoder.load_state_dict(torch.load(path))
    

    def encode_visual_information(self, x: TensorType["batch", "(C, H, W)"]):
        assert callable(self._visual_encoder.encoder), "The class has not the encoder function"
        embedding = self._visual_encoder.encoder(x)
        
        return embedding
    
    
    def encode_edge_positional_information(self, 
                                           x: TensorType["batch", "d"]) -> TensorType["batch", "|E|", "d"]:
        
        messages = []
        for edge_module in (self._edge_encoders):
            message = edge_module(x)
            messages.append(message.unsqueeze(1))
        
        messages = torch.cat(messages, dim=1)  
        
        return messages
    
    
    def encode_attribute_information(self,image_features:[TensorType["batch", "d"]],  # type: ignore
                edge_attributes: [TensorType["batch", "Nattributes", "d"]]) -> TensorType["batch", "|E|", "d"]:         # type: ignore
        

        return self._gnn_encoder(image_features=image_features, edge_attributes=edge_attributes)
    
            
    def forward(self, x: TensorType["batch", "(C, H, W)"]):

        image_features = self.encode_visual_information(x)
        edge_features = self.encode_edge_positional_information(image_features)

        attribute_representation, individual_embeddings = self.encode_attribute_information(image_features, edge_features)

        return attribute_representation, individual_embeddings
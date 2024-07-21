
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


        self._visual_encoder = visual_encoder(cfg=cfg.line_encoder)
        self._gnn_encoder = gnn_encoder(cfg=cfg.gnn_encoder)
        self._apply_edges = cfg.apply_edge_encoder

        if self._apply_edges is not False:
            self._edge_encoder = edge_encoder(cfg = cfg.edge_encoder)


    def encode_visual_information(self, x: TensorType["batch", "(C, H, W)"]):
        return self._visual_encoder(x)
    
    
    def encode_edge_positional_information(self, x: TensorType["batch", "(C, H, W)"]):
        return self._edge_encoder(x)
    
    
    def encode_attribute_information(self, x_dict: Dict[str, TensorType],
                edge_index_dict: Dict[Tuple[str, str, str], Adj],
                population: TensorType["batch_individual_nodes"],
                edge_attributes: Optional[TensorType["batch", "Nattributes", "embedding_size"]] = None
    ):
        
        if edge_attributes is None:
            return self._gnn_encoder(x_dict=x_dict, edge_index_dict=edge_index_dict, population=population)    
        else:
            return self._gnn_encoder(x_dict=x_dict, edge_index_dict=edge_index_dict, edge_attributes=edge_attributes, population=population)
    


            

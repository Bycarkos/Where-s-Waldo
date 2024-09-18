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
    
    def __init__(self, cfg: DictConfig):
          
        super().__init__()
        
        self.node_types = cfg.node_types
        self.embedding_size = cfg.embedding_size
        self._activation = nn.ReLU(inplace=False)

        self._edge_space_projection = nn.Parameter(data=torch.randn(size=(len(self.node_types), self.embedding_size, self.embedding_size)), requires_grad=True).to(device=device)
        nn.init.kaiming_uniform_(self._edge_space_projection)

        self._edge_space_aggregator = nn.Linear(in_features=2*self.embedding_size, out_features=self.embedding_size)

    
    def compute_edge_message(self, src_embeddings, positional_features):
        
        src_embeddings = src_embeddings.unsqueeze(1).repeat(1, positional_features.shape[1],1)
        individual_edge_information = torch.cat((src_embeddings, positional_features), dim=-1)
        aggregated_information = self._edge_space_aggregator(individual_edge_information) #(B, selg._attributes, embeddingsize )
        edge_space_projector = torch.einsum('bai,aij->baj', aggregated_information, self._edge_space_projection)

        return edge_space_projector

    def forward(self,image_features:[TensorType["batch", "embedding_size"]],
                edge_attributes: [TensorType["batch", "Nattributes", "embedding_size"]] 
):


        forward_message = self.compute_edge_message(src_embeddings=image_features, positional_features=edge_attributes)
        

        return forward_message     
    
    

class BackwardAttributeGnn(nn.Module):
    
    
    def __init__(self, cfg: DictConfig):
          
        super().__init__()
        
        self.node_types = cfg.node_types
        self.embedding_size = cfg.embedding_size
        self._activation = nn.ReLU(inplace=False)
        self._edge_space_projection = nn.Parameter(data=torch.randn(size=(len(self.node_types), self.embedding_size, self.embedding_size)), requires_grad=True).to(device=device)
        self._self_weighted = nn.Parameter(data=torch.ones(size=(1, len(self.node_types), 1)), requires_grad=True).to(device=device)
        nn.init.kaiming_uniform_(self._edge_space_projection)


    def compute_semantic_message(self, attribute_embeddings: TensorType["batch", "attribute_nodes", "embedding_size"]):
        
        
        #[b a, i]
        #[a, i, j ]
        edge_space_projector = torch.einsum('bai,aij->baj', attribute_embeddings, self._edge_space_projection)
        edge_space_projector = self._activation(edge_space_projector)
        weighted_embeddings = torch.sum(edge_space_projector * self._self_weighted, dim=1)
        return weighted_embeddings
    
        
    def forward(self, attribute_embeddings: TensorType["n_attributes", "total_nodes", "embedding_size"]):
        
        embeddings_individuals = self.compute_semantic_message(attribute_embeddings=attribute_embeddings)

        return embeddings_individuals  
    
    
    
class AttributeGNN(nn.Module):
    
    def __init__(self, cfg:DictConfig) -> None:
        super().__init__()
        
        self._forward_attribute_message = ForwardAttributeGnn(cfg.forward_attribute_gnn_encoder)
        self._backward_attribute_message = BackwardAttributeGnn(cfg.backward_attribute_gnn_encoder)
        
        self._apply_backward = cfg.add_backward
    
    def forward(self, image_features:[TensorType["batch", "embedding_size"]],
                edge_attributes: [TensorType["batch", "Nattributes", "embedding_size"]]):
                        
        attribute_representation = self._forward_attribute_message(image_features,  edge_attributes)
        
        if self._apply_backward:
            individual_embeddings = self._backward_attribute_message(attribute_representation)
            
            
            return attribute_representation, individual_embeddings

        else:
            return attribute_representation, None
        
        
     
  
class FamilyAttributeGnn(nn.Module):

    def __init__(self, cfg: DictConfig):
          
        super().__init__()
        attribute_edges =  [("individuals", etype, etype) for etype in cfg.attributes]
        self._attributes = cfg.attributes

        ### Attributes
        self._attributes_update = nn.ModuleList()
        self._attr_ind_convs = nn.ModuleList()
        self._ind_ind_convs = nn.ModuleList()
        self._activation = nn.ReLU(inplace=False)
        
        # ^ Primer enviam missatges del individu al attrbut 
        ind_attribute_conv = {attrribute_etype: SAGEConv((-1, -1), cfg.embedding_size, flow="source_to_target") for attrribute_etype in attribute_edges}
        self.direct_conv = HeteroConv(ind_attribute_conv, aggr=cfg.aggr)

        # ^ Ara actualitzem la informació dels individus i tornem a enviar missatges
        ind_ind_conv = {("individuals", "family", "individuals"): SAGEConv((-1, -1), cfg.embedding_size, flow="target_to_source")}
        self.individual_message = HeteroConv(ind_ind_conv)
        
        # ^ Amb les informacions actualitzades en paral·lel tornem a enviar els missatges cap als attributs
        ind_ind_conv = {("individuals", "family", "individuals"): SAGEConv((-1, -1), cfg.embedding_size, flow="source_to_target")}
        ind_attribute_conv = {attrribute_etype: SAGEConv((-1, -1), cfg.embedding_size, flow="source_to_target") for attrribute_etype in attribute_edges}
        ind_attribute_conv.update(ind_ind_conv)
        self._inverse_direct = HeteroConv(ind_attribute_conv, aggr=cfg.aggr)
        

        self._edge_space_projection = {att: nn.Parameter(data=torch.randn(size=(cfg.embedding_size, cfg.embedding_size)), requires_grad=True).to(device=device) for att in self._attributes}   
        nn.init.kaiming_uniform_({key: nn.init.kaiming_uniform_(values, a=0.2) for key, values in self._edge_space_projection.items()})

        
        self._edge_space_aggregator = nn.Linear(in_features=2*cfg.embedding_size, out_features=cfg.embedding_size)


    def apply_edges_on_attributes(self, x_dict: Dict[str, TensorType],
                                  positional_features: TensorType["batch", "Nattributes", "embedding_size"], 
                                  population: TensorType["batch_individual_nodes"]):


        """
        Apply edge attributes to the features of nodes (individuals) using the attributes.

        Parameters:
        -----------
        x_dict : dict
            A dictionary containing the feature matrices of the nodes. 
            The keys are the node types (e.g., "individuals", attribute types), 
            and the values are the corresponding feature matrices of shape [num_nodes, num_features].

        positional_features : torch.Tensor
            A tensor containing the positional features of the edges.
            Shape: [num_edges, num_attributes, embedding_size].

        population : list or torch.Tensor
            A list or tensor containing the indices of the individuals in the current population.

        Returns:
        --------
        x_dict : dict
            A dictionary containing the updated feature matrices of the nodes after applying edge attributes.
        """

        for idx, attribute in enumerate(self._attributes):
            edge_space_features_positional = (positional_features[:, idx, :] @ self._edge_space_projection[attribute]) 
            attributte_information = x_dict[attribute][population] @ self._edge_space_projection[attribute] 
            aggregation = torch.cat((attributte_information, edge_space_features_positional.cuda()), dim=1).cuda()
            aggregated_information = self._edge_space_aggregator(aggregation)

            x_dict[attribute][population] = aggregated_information


        return x_dict

    def forward(self, x_dict: Dict[str, TensorType],
                edge_index_dict: Dict[Tuple[str, str, str], Adj],
                population: TensorType["batch_individual_nodes"],
                edge_attributes: Optional[TensorType["batch", "Nattributes", "embedding_size"]] = None, 
):
        

        
        attribute_dict = self.direct_conv(x_dict, edge_index_dict)
        attribute_dict = {key: x.relu() for key, x in attribute_dict.items()}
       
        individual_dict = self.individual_message(x_dict, edge_index_dict)
        individual_dict = {key: x.relu() for key, x in individual_dict.items()}
        
        x_dict.update(attribute_dict)
        x_dict.update(individual_dict)
        
        x_dict = self._inverse_direct(x_dict, edge_index_dict)

        if edge_attributes is not None:
            x_dict = self.apply_edges_on_attributes(x_dict=x_dict, positional_features=edge_attributes, population=population)
 
        return x_dict     
    

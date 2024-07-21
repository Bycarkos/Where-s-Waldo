import torch.nn as nn
import torch
from torch_geometric.nn import HGTConv, HeteroConv, GCNConv, SAGEConv, GATConv, Linear

from torch_geometric.typing import Adj
from torchtyping import TensorType
from typing import *


from omegaconf import DictConfig, OmegaConf


device = "cuda" if torch.cuda.is_available() else "cpu"



class AttributeGnn(nn.Module):
    
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
        self.conv = HeteroConv(ind_attribute_conv, aggr=cfg.aggr)


        self._edge_space_projection = {att: nn.Parameter(data=torch.randn(size=(cfg.embedding_size, cfg.embedding_size)), requires_grad=True).to(device=device) for att in self._attributes}   
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
        

        
        x_dict = self.conv(x_dict, edge_index_dict)
        if edge_attributes is not None:
            x_dict = self.apply_edges_on_attributes(x_dict=x_dict, positional_features=edge_attributes, population=population)
 
        return x_dict     
    
    
     
  
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

        #x_dict = {key: x.relu() for key, x in x_dict.items()}
        if edge_attributes is not None:
            x_dict = self.apply_edges_on_attributes(x_dict=x_dict, positional_features=edge_attributes, population=population)
 
        return x_dict     
    

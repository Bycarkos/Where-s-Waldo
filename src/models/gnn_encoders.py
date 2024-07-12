import torch.nn as nn
import torch
from torch_geometric.nn import HGTConv, HeteroConv, GCNConv, SAGEConv, GATConv, Linear

from torch_geometric.typing import Adj
from torchtyping import TensorType
from typing import *


from omegaconf import DictConfig, OmegaConf


device = "cuda" if torch.cuda.is_available() else "cpu"

    
class HeteroGNN(nn.Module):
    def __init__(self, cfg: DictConfig):

        """
        Initialize the Heterogeneous Graph Neural Network (HeteroGNN).

        Parameters:
        -----------
        attributes : list
            A list of attribute types used in the graph.
        
        embedding_size : int, optional
            The size of the embeddings for the attributes, by default 128.
        
        aggr : str, optional
            The aggregation method to use in HeteroConv layers, by default "mean".
        """
        

        super().__init__()

        attribute_edges =  [("individuals", etype, etype) for etype in cfg.attributes]
        self._attributes = cfg.attributes

        ### Attributes
        self._ind_attribute_convs = nn.ModuleList()
        self._attr_ind_convs = nn.ModuleList()
        self._ind_ind_convs = nn.ModuleList()

        conv = HeteroConv({
            attrribute_etype: SAGEConv((-1, -1), cfg.embedding_size, flow="source_to_target") for attrribute_etype in attribute_edges
        }, aggr=cfg.aggr)

        self._ind_attribute_convs.append(conv)
        
        
        self._activation = nn.ReLU(inplace=False)
        
        ### Attributes to individuals 
        conv = HeteroConv({
            attrribute_etype: SAGEConv((-1, -1), cfg.embedding_size, flow="target_to_source") for attrribute_etype in attribute_edges
        }, aggr=cfg.aggr)
        
        self._attr_ind_convs.append(conv)


        ### Individuals to Individuals
        conv = HeteroConv({
            ("individuals", "family", "individuals"): SAGEConv((-1, -1), cfg.embedding_size, flow="source_to_target"),
            ("individuals", "family", "individuals"): SAGEConv((-1, -1), cfg.embedding_size, flow="target_to_source")
        }, aggr=cfg.aggr)

        self._ind_ind_convs.append(conv)


        self._edge_space_projection = {att: nn.Parameter(data=torch.randn(size=(cfg.embedding_size, cfg.embedding_size)), requires_grad=True).to(device=device) for att in self._attributes}
        
        self._edge_space_aggregator = nn.Linear(in_features=2*cfg.embedding_size, out_features=cfg.embedding_size)
        self._identity = torch.eye(cfg.embedding_size).to(device=device)

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

            edge_space_features = (positional_features[:, idx, :] @ self._edge_space_projection[attribute]) 
            x_dict[attribute] = x_dict[attribute].cuda()
            aggregation = torch.cat((x_dict[attribute][population], edge_space_features.cuda()), dim=1).cuda()
            aggregated_information = self._edge_space_aggregator(aggregation)

            x_dict[attribute][population] = aggregated_information
            x_dict[attribute][population] = self._activation(aggregated_information)


        return x_dict


    def apply_edges_on_individuals(self, x_dict:Dict[str, TensorType],
                                    population: TensorType["batch_individual_nodes"]):
            
        """
            Aggregate attribute information back into individuals.

            Parameters:
            -----------
            x_dict : dict
                A dictionary containing the feature matrices of the nodes. 
                The keys are the node types (e.g., "individuals", attribute types), 
                and the values are the corresponding feature matrices of shape [num_nodes, num_features].

            population : list or torch.Tensor
                A list or tensor containing the indices of the individuals in the current population.

            Returns:
            --------
            x_dict : dict
                A dictionary containing the updated feature matrices of the nodes after aggregating attribute information.
        """

        for idx, attribute in enumerate(self._attributes):
            inverse_edge_space = x_dict[attribute][population] @ (torch.inverse(self._edge_space_projection[attribute]) + self._identity)
            x_dict[attribute][population] = inverse_edge_space
            x_dict[attribute][population] = self._activation(inverse_edge_space)


        return x_dict

## TODO SOlucionar el forward per que hi ha probnlemes amb el backpropagation. ( i think is solve4d)

    def forward(self, x_dict: Dict[str, TensorType],
                edge_index_dict: Dict[Tuple[str, str, str], Adj],
                edge_attributes: TensorType["batch", "Nattributes", "embedding_size"], 
                population: TensorType["batch_individual_nodes"]):


        """
        Forward pass for the Heterogeneous Graph Neural Network (HeteroGNN).

        Parameters:
        -----------
        x_dict : dict
            A dictionary containing the feature matrices of the nodes. 
            The keys are the node types (e.g., "individuals", attribute types), 
            and the values are the corresponding feature matrices of shape [num_nodes, num_features].

        edge_index_dict : dict
            A dictionary containing the edge indices for each edge type.
            The keys are tuples representing the edge types (e.g., ("individuals", "attribute_type", "individuals")), 
            and the values are the corresponding edge index tensors of shape [2, num_edges].

        edge_attribute_dict : torch.Tensor
            A tensor containing the edge attributes/features for the edges between individuals and attributes.
            Shape: [num_edges, num_attributes, embedding_size].

        population : list or torch.Tensor
            A list or tensor containing the indices of the individuals in the current population.

        Returns:
        --------
        x_dict : dict
            A dictionary containing the updated feature matrices of the nodes after the forward pass.
            The keys are the node types (e.g., "individuals", attribute types), 
            and the values are the corresponding updated feature matrices of shape [num_nodes, num_features].

        Description:
        ------------
        The forward pass consists of three main stages:
        
        1. Applying edge attributes to the features of nodes (individuals) using the attributes:
            - For each attribute type, positional features are transformed and concatenated with node features.
            - The concatenated features are passed through a linear layer to get aggregated information.
            - This updated information is used to modify the node features of the attributes in `x_dict`.
        
        2. Aggregating attribute information back into individuals:
            - For each attribute type, the inverse of the edge space projection matrix is applied to the attribute features.
            - This inverse transformed information is used to update the individual features in `x_dict`.
        
        3. Applying convolution operations to update the node features:
            - For each convolution layer in `_ind_attribute_convs`, `_attr_ind_convs`, and `_ind_ind_convs`, 
            the respective convolutions are applied to the `x_dict` using the `edge_index_dict`.
            - After each convolution, ReLU activation is applied to the updated node features.
        
        The final updated `x_dict` contains the new node features after applying the heterogeneous graph convolutions.
        """


        for convs in self._ind_attribute_convs:
            x_dict = self.apply_edges_on_attributes(x_dict=x_dict, positional_features=edge_attributes, population=population)
            x_dict_attributes = convs(x_dict, edge_index_dict)

        x_dict.update(x_dict_attributes)

        for convs in self._attr_ind_convs:
            x_dict = self.apply_edges_on_individuals(x_dict=x_dict, population=population)
            x_dict_individuals = convs(x_dict, edge_index_dict)

        x_dict.update(x_dict_individuals)
        

        for conv in self._ind_ind_convs:
            x_dict_family = conv(x_dict=x_dict, edge_index_dict=edge_index_dict)

        x_dict.update(x_dict_family)
 

        return x_dict

    
    
   
## ** Example 
class HGT(nn.Module):
    def __init__(self, data, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()
        
        self.lin_dict = nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict['author'])
import torch.nn as nn
import torch

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


# ^ make it configurable when all is done 
class LineFeatureExtractor(nn.Module):
    
    def __init__(self, min_height: int, max_width: int, hidden_channels:int, output_channels:int, num_hidden_convolutions:int=3) -> None:
        super(LineFeatureExtractor, self).__init__()
        
        self._min_heigh = min_height
        self._max_width = max_width

        self._hidden_channels = hidden_channels
        self._output_channels = output_channels

        self._pool = nn.AdaptiveAvgPool2d(1)

        self._num_middle_conv = num_hidden_convolutions - 2

        self._list_convolutions = []

        self._list_convolutions.append(nn.Conv2d(in_channels=3, out_channels=hidden_channels, kernel_size=(min_height, max_width), device="cuda"))

        for i in range(self._num_middle_conv):
            self._list_convolutions.append(nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=(3, 5), device="cuda"))

        self._list_convolutions.append(nn.Conv2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=(3, 5), device="cuda"))

        
    def forward(self, x):
        for conv in self._list_convolutions:
            x = conv(x)
            x = torch.relu(x)
            
        x = self._pool(x).view(x.shape[0], -1)

        return x

class EdgeAttFeatureExtractor(LineFeatureExtractor):
    
    def __init__(self, min_height: int, max_width: int, hidden_channels:int, output_channels:int, number_of_entities:int=5, edge_embedding_size:int=128 ) -> None:
        super(EdgeAttFeatureExtractor, self).__init__(max_width=max_width, min_height=min_height, hidden_channels=hidden_channels, output_channels=output_channels, num_hidden_convolutions=number_of_entities)
        
        self._conv1 = nn.Conv2d(in_channels=3, out_channels=hidden_channels, kernel_size=(min_height, max_width))
        self._conv2 = nn.Conv2d(in_channels=hidden_channels, out_channels=5, kernel_size=(3, 5))
        self._edge_projector_pooling = nn.AdaptiveAvgPool2d(10) ## image which will be with embedding 100
        
        self._edge_projector = nn.Linear(10**2, out_features=edge_embedding_size)
        self._number_of_entities = number_of_entities

    def forward(self, x):
        x = self._conv1(x)
        x = torch.relu(x)
        x = self._conv2(x)

        x = self._edge_projector_pooling(x).view(x.shape[0],self._number_of_entities,  -1)
        x = torch.relu(x)
        x = self._edge_projector(x)
        x = torch.relu(x)

        return x



class LineVITFeatureExtractor(nn.Module):

    def __init__(self, cfg: DictConfig ) -> None:
        super(LineVITFeatureExtractor).__init__()
        
        self._encoder = instantiate(cfg.encoder)


    def forward(self, x):

        return torch.relu(self._encoder(x))
        
        

class DocumentFeatureExtractor(nn.Module):
    
    def __init__(self) -> None:
        super(DocumentFeatureExtractor, self).__init__()

        
        
    def forward(self):
        raise NotImplementedError
        
        
        


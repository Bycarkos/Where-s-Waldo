import torch.nn as nn
import torch

from hydra.utils import instantiate
from omegaconf import DictConfig

device = "cuda" if torch.cuda.is_available else "cpu"

# ^ make it configurable when all is done 

#    min_height=kernel_height, max_width=kernel_width, hidden_channels=64, output_channels=128, num_hidden_convolutions=3

class LineFeatureExtractor(nn.Module):
    
    def __init__(self, cfg:DictConfig) -> None:
        super(LineFeatureExtractor, self).__init__()
        
        self._min_heigh = cfg.kernel_height
        self._max_width = cfg.kernel_width

        self._hidden_channels = cfg.hidden_channels
        self._output_channels = cfg.output_channels

        self._pool = nn.AdaptiveAvgPool2d(1)

        self._num_middle_conv = cfg.num_hidden_convolutions - 2

        self._list_convolutions = []

        self._list_convolutions.append(nn.Conv2d(in_channels=3, out_channels=self._hidden_channels, kernel_size=(self._min_heigh, self._max_width), device=device))

        for i in range(self._num_middle_conv):
            self._list_convolutions.append(nn.Conv2d(in_channels=self._hidden_channels, out_channels=self._hidden_channels, kernel_size=(5, 3), device=device))

        self._list_convolutions.append(nn.Conv2d(in_channels=self._hidden_channels, out_channels=self._output_channels, kernel_size=(5, 3), device=device))

        
    def forward(self, x):
        for conv in self._list_convolutions:
            x = conv(x)
            x = torch.relu(x)
            
        x = self._pool(x).view(x.shape[0], -1)

        return x

#min_height: int, max_width: int, hidden_channels:int, output_channels:int, number_of_entities:int=5, edge_embedding_size:int=128
class EdgeAttFeatureExtractor(nn.Module):
    
    def __init__(self, cfg: DictConfig ) -> None:
        super(EdgeAttFeatureExtractor, self).__init__()
        
        self._min_heigh = cfg.kernel_height
        self._max_width = cfg.kernel_width
        
        self._hidden_channels = cfg.hidden_channels
        self._output_channels = cfg.number_of_entities    

        self._embedding_size = cfg.edge_embedding_size   

        self._conv1 = nn.Conv2d(in_channels=3, out_channels=self._hidden_channels, kernel_size=(self._min_heigh, self._max_width))
        self._conv2 = nn.Conv2d(in_channels=self._hidden_channels, out_channels=self._output_channels, kernel_size=(3, 5))
        self._edge_projector_pooling = nn.AdaptiveAvgPool2d(10) ## image which will be with embedding 100
        
        self._edge_projector = nn.Linear(10**2, out_features=self._embedding_size)

    def forward(self, x):
        x = self._conv1(x)
        x = torch.relu(x)
        x = self._conv2(x)

        x = self._edge_projector_pooling(x).view(x.shape[0],self._output_channels,  -1)
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
        
        
        


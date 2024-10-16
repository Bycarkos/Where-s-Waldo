
import torch.nn as nn
import torch

import models.attentions as att

import math
from hydra.utils import instantiate
from omegaconf import DictConfig

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from torchtyping import TensorType

import pdb

device = "cuda" if torch.cuda.is_available else "cpu"

from transformers import TrOCRProcessor


class EdgeTokenizerFeatureExtractor(nn.Module):

    def __init__(self, cfg: DictConfig) -> None:
        super(EdgeTokenizerFeatureExtractor, self).__init__()

        self._patch_size = cfg.patch_size
        self._output_channels = len(cfg.number_of_different_edges)  
        self._embedding_size = cfg.output_channels   

        
        ## Positional Encoding
        self._positional_max_length = cfg.positional_max_length
        self.pe = att.PositionalEncoding(d_model=self._patch_size, max_len=self._positional_max_length)

        
        ## Attention Mechanism
        self._add_attention = cfg.add_attention


        self._input_attention_mechanism = cfg.input_atention_mechanism * 3

        self._patch_to_embedding = nn.Sequential(
            Rearrange( 'b c h (w p) -> b (p) (w h c)', p=self._patch_size),
            nn.LayerNorm(self._input_attention_mechanism),
            nn.Linear(self._input_attention_mechanism, self._embedding_size),
            nn.LayerNorm(self._embedding_size),
        )

        self._scaled_attention = att.ScaledDotProductAttention()
        self._Q = nn.Linear(self._embedding_size, self._embedding_size)
        self._K = nn.Linear(self._embedding_size, self._embedding_size)
        self._V = nn.Linear(self._embedding_size, self._embedding_size)      
        self._scaled_attention = att.ScaledDotProductAttention()

        self.cls_token = nn.Parameter(torch.randn(1, self._output_channels, self._embedding_size))

        ## embedding_projection
        self._output_layer = nn.Linear(self._input_attention_mechanism, self._embedding_size)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_positioned = self.pe(x.view(B, self._patch_size, C*H*W//self._patch_size))
        x_positioned = x_positioned.view(B, C, H, W)
        x = self._patch_to_embedding(x_positioned)

        cls_tokens = repeat(self.cls_token, '1 att d -> b att d', b = B, att=self._output_channels)
        x = torch.cat((cls_tokens, x), dim=1)

        if self._add_attention:
            queries = self._Q(x)
            keys = self._K(x)
            values = self._V(x)

            x, self.attention_values = self._scaled_attention(query=queries,
                                                                    key=keys,
                                                                    value=values)
        
        return x[:, :3]

## Problema d'aquest, el visual encoder si que està capturant el overall de les shapes, pero aquest no
#min_height: int, max_width: int, hidden_channels:int, output_channels:int, number_of_entities:int=5, edge_embedding_size:int=128
class EdgeAttFeatureExtractor(nn.Module):
    
    def __init__(self, cfg: DictConfig) -> None:
        super(EdgeAttFeatureExtractor, self).__init__()
        
        
        self._height = cfg.kernel_height
        self._width = cfg.kernel_width
        self._patch_size = cfg.patch_size
        
        
        self._kernels = list(zip(self._height, self._width))
        self._input_channels = cfg.input_channels
        self._hidden_channels = cfg.hidden_channels
        self._output_channels = len(cfg.number_of_different_edges)  

        self._embedding_size = cfg.output_channels   
                                                                                    ## Amb tots els volums (10, 3), amb menys volums (5 ,3)
        self._conv1 = nn.Conv2d(in_channels=self._input_channels, out_channels=self._input_channels, kernel_size=self._kernels[0], padding=(2,0), stride=(1, self._patch_size))
        self._conv2 = nn.Conv2d(in_channels=self._input_channels, out_channels=self._hidden_channels, kernel_size=(3, 3), padding="same")
        self._conv3 = nn.Conv2d(in_channels=self._hidden_channels, out_channels=self._output_channels, kernel_size=(3, 3), padding="same")
        
        self._add_attention = cfg.add_attention
        self._positional_max_length = cfg.positional_max_length
        self._input_attention_mechanism = cfg.input_atention_mechanism

        self._output_layer = nn.Linear(self._input_attention_mechanism, self._embedding_size)
        
        self.pe = att.PositionalEncoding(d_model=self._patch_size, max_len=self._positional_max_length)
        self._Q = nn.Linear(self._input_attention_mechanism, self._embedding_size)
        self._K = nn.Linear(self._input_attention_mechanism, self._embedding_size)
        self._V = nn.Linear(self._input_attention_mechanism, self._embedding_size) 
        self._scaled_attention = att.ScaledDotProductAttention()
        
        self._to_patch = Rearrange( 'b c h (w p) -> b (p) (w h c)', p=self._patch_size)

    def forward(self, x):
        ## first we add the positional encoding
         
        x_patched = self._to_patch(x) # output = B, 16, 3 * 44 * 48
        x_patched = self.pe(x_patched)
        x = x_patched.view(*x.shape)
        
        x = self._conv1(x) # output = B, 3, 44, 48
        x = torch.relu(x)
        x = self._conv2(x)
        x = torch.relu(x)
        x = self._conv3(x)

        x = x.view(x.shape[0], x.shape[1], -1)

        if self._add_attention:
            queries = self._Q(x)
            keys = self._K(x)
            values = self._V(x)
            

            x, self.attention_values = self._scaled_attention(query=queries,
                                                                    key=keys,
                                                                    value=values)
            
        else:    
            x = self._output_layer(x)
            
        return x
    



class DisentanglementAttentionEncoder(nn.Module):

    def __init__(self, in_features:int, out_features:int) -> None:
        
        super(DisentanglementAttentionEncoder, self).__init__()

        self._in_features = in_features
        self._out_features= out_features

        self._Q = nn.Linear(in_features=self._in_features, out_features=self._out_features)
        self._K = nn.Linear(in_features=self._in_features, out_features=self._out_features)
        self._V = nn.Linear(in_features=self._in_features, out_features=self._out_features)

        self._attention_mechanism = att.ScaledDotProductAttention()


    def forward(self, embeddings: TensorType["Batch", "d"]) -> TensorType["embedding", "d"]:

        queries = self._Q(embeddings)
        keys = self._K(embeddings)
        values = self._V(embeddings)

        x, self._attention_values = self._attention_mechanism(query=queries[:,:, None],
                                                              key=keys[:,:, None],
                                                              value=values[:,:, None])
        return x.squeeze(-1)
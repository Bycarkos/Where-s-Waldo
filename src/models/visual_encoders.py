import torch.nn as nn
import torch


import math
from hydra.utils import instantiate
from omegaconf import DictConfig

import pdb

device = "cuda" if torch.cuda.is_available else "cpu"

from transformers import TrOCRProcessor

#    min_height=kernel_height, max_width=kernel_width, hidden_channels=64, output_channels=128, num_hidden_convolutions=3

class LineFeatureExtractor(nn.Module):
    
    def __init__(self, cfg:DictConfig) -> None:
        super(LineFeatureExtractor, self).__init__()
        
        self._height = cfg.kernel_height
        self._width = cfg.kernel_width
        
        self._kernels = list(zip(self._height, self._width))

        self._input_channels = cfg.input_channels
        self._hidden_channels = cfg.hidden_channels
        self._output_channels = cfg.output_channels

        self._pool = nn.AdaptiveAvgPool2d(1)

        self._num_middle_conv = cfg.number_of_hidden_convolutions

        self._list_convolutions = []

        self._list_convolutions.append(nn.Conv2d(in_channels=3, out_channels=self._hidden_channels, kernel_size=self._kernels[0], device=device))#(self._min_heigh, self._max_width), device=device))

        for i in range(self._num_middle_conv):
            self._list_convolutions.append(nn.Conv2d(in_channels=self._hidden_channels, out_channels=self._hidden_channels, kernel_size=self._kernels[i+1], device=device))

        self._list_convolutions.append(nn.Conv2d(in_channels=self._hidden_channels, out_channels=self._output_channels, kernel_size=self._kernels[-1], device=device))

        
    def forward(self, x):
        for conv in self._list_convolutions:
            x = conv(x)
            x = torch.relu(x)
            
        x = self._pool(x).view(x.shape[0], -1)

        return x
    



class LineVariationalAutoEncoder(nn.Module):
    def __init__(self, cfg:DictConfig):

    
        super(LineAutoEncoder, self).__init__()

        self._height = cfg.kernel_height
        self._width = cfg.kernel_width
        
        self._kernels = list(zip(self._height, self._width))

        self._input_channels = cfg.input_channels
        self._hidden_channels = cfg.hidden_channels
        self._output_channels = cfg.output_channels

        self._num_middle_conv = cfg.number_of_hidden_convolutions

        if self._num_middle_conv == 1:
            self._norm = nn.BatchNorm2d(self._hidden_channels)
        else:
            self._norm = nn.BatchNorm2d(self._hidden_channels * (self._num_middle_conv+1))
        

        self._list_convolutions = []
        self._list_deconvolutions = []
        self._list_pooling = []

        
        self._initialize_encoder()
        self._initialize_decoder()
        
    def _initialize_encoder(self):
        
        self._max_pooling = nn.MaxPool2d(2, stride=2, padding=1, return_indices=True)

        ## initialize encoder
        self._init_convolution = nn.Conv2d(in_channels=3, out_channels=self._hidden_channels, kernel_size=(5,3), device=device)
        if self._num_middle_conv == 1:
            self._list_convolutions.append(nn.Conv2d(in_channels=(self._hidden_channels), out_channels=(self._hidden_channels), kernel_size=(5,3), device=device))

        else:
            for i in range(1, self._num_middle_conv + 1):
                self._list_convolutions.append(nn.Conv2d(in_channels=(self._hidden_channels * i), out_channels=(self._hidden_channels * (i+1)), kernel_size=(5,3), device=device))

        if self._num_middle_conv == 1:
            self._output_convolution = nn.Conv2d(in_channels=self._hidden_channels , out_channels=self._output_channels, kernel_size=(5,3), device=device)
        else:
            self._output_convolution = nn.Conv2d(in_channels=self._hidden_channels * (self._num_middle_conv+1), out_channels=self._output_channels, kernel_size=(5,3), device=device)

    def _initialize_decoder(self):
        self._max_unpooling = nn.MaxUnpool2d(2, stride=2, padding=1)

        if self._num_middle_conv == 1:
            self._output_deconvolution = nn.ConvTranspose2d(in_channels=self._output_channels, out_channels=self._hidden_channels, kernel_size=(5,3), device=device)
        else:
            self._output_deconvolution = nn.ConvTranspose2d(in_channels=self._output_channels, out_channels=self._hidden_channels * (self._num_middle_conv+1), kernel_size=(5,3), device=device)

        if self._num_middle_conv == 1:
            self._list_deconvolutions.append(nn.ConvTranspose2d(in_channels=(self._hidden_channels), out_channels=(self._hidden_channels), kernel_size=(5,3), device=device))
        else:
            for i in range(self._num_middle_conv + 1, 1, -1):
                self._list_deconvolutions.append(nn.ConvTranspose2d(in_channels=(self._hidden_channels * i), out_channels=(self._hidden_channels * (i-1)), kernel_size=(5,3), device=device))

        self._init_deconvolution = nn.ConvTranspose2d(in_channels=self._hidden_channels, out_channels=3, kernel_size=(5,3), device=device)



class LineAutoEncoder(nn.Module):


    def __init__(self, cfg:DictConfig):


        super(LineAutoEncoder, self).__init__()

        self._height = cfg.kernel_height
        self._width = cfg.kernel_width
        
        self._kernels = list(zip(self._height, self._width))

        self._input_channels = cfg.input_channels
        self._hidden_channels = cfg.hidden_channels
        self._output_channels = cfg.output_channels

        self._num_middle_conv = cfg.number_of_hidden_convolutions

        if self._num_middle_conv == 1:
            self._norm = nn.BatchNorm2d(self._hidden_channels)
        else:
            self._norm = nn.BatchNorm2d(self._hidden_channels * (self._num_middle_conv+1))
        

        self._list_convolutions = []
        self._list_deconvolutions = []
        self._list_pooling = []

        

        self._initialize_encoder()
        self._initialize_decoder()


    def _initialize_encoder(self):
        
        self._max_pooling = nn.MaxPool2d(2, stride=2, padding=1, return_indices=True)

        ## initialize encoder
        self._init_convolution = nn.Conv2d(in_channels=3, out_channels=self._hidden_channels, kernel_size=(5,5), device=device)
        if self._num_middle_conv == 1:
            self._list_convolutions.append(nn.Conv2d(in_channels=(self._hidden_channels), out_channels=(self._hidden_channels), kernel_size=(5,5), device=device))

        else:
            for i in range(1, self._num_middle_conv + 1):
                self._list_convolutions.append(nn.Conv2d(in_channels=(self._hidden_channels * i), out_channels=(self._hidden_channels * (i+1)), kernel_size=(5,5), device=device))

        if self._num_middle_conv == 1:
            self._output_convolution = nn.Conv2d(in_channels=self._hidden_channels , out_channels=self._output_channels, kernel_size=(5,5), device=device)
        else:
            self._output_convolution = nn.Conv2d(in_channels=self._hidden_channels * (self._num_middle_conv+1), out_channels=self._output_channels, kernel_size=(5,5), device=device)


    def _initialize_decoder(self):
        self._max_unpooling = nn.MaxUnpool2d(2, stride=2, padding=1)
        self._global_max_unpool = nn.MaxUnpool2d(4)
        if self._num_middle_conv == 1:
            self._output_deconvolution = nn.ConvTranspose2d(in_channels=self._output_channels, out_channels=self._hidden_channels, kernel_size=(5,5), device=device)
        else:
            self._output_deconvolution = nn.ConvTranspose2d(in_channels=self._output_channels, out_channels=self._hidden_channels * (self._num_middle_conv+1), kernel_size=(5,5), device=device)

        if self._num_middle_conv == 1:
            self._list_deconvolutions.append(nn.ConvTranspose2d(in_channels=(self._hidden_channels), out_channels=(self._hidden_channels), kernel_size=(5,5), device=device))
        else:
            for i in range(self._num_middle_conv + 1, 1, -1):
                self._list_deconvolutions.append(nn.ConvTranspose2d(in_channels=(self._hidden_channels * i), out_channels=(self._hidden_channels * (i-1)), kernel_size=(5,5), device=device))

        self._init_deconvolution = nn.ConvTranspose2d(in_channels=self._hidden_channels, out_channels=3, kernel_size=(5,5), device=device)


    def encoder(self, x):

        x = self._init_convolution(x)
        x = torch.relu(x)

        ## getting the exepcted shape because of rounding     
        # As explained in the docs for MaxUnpool: when doing MaxPooling, there might be some pixels that get rounded up due to integer division on the input size.
        # For example, if your image has size 5, and your stride is 2, the output size can be either 2 or 3, and you can’t retrieve the original size of the image. 
        # That’s why there is an optional argument to MaxUnpool that allows you to specify the desired output size. In your case, it should be something like  

        self._pooling_shape = x.shape
        x, self._indices = self._max_pooling(x)
        
        for conv in self._list_convolutions:
            x = conv(x)
            x = torch.relu(x)

        self.__x_init_skip_connections = x
        
        x = self._norm(x)
        x = self._output_convolution(x)
        
        
        self.__x_skip_connection = x
        self._encode_recover_shape = x.shape
        self._final_pooling_shape = x.view(x.shape[0], x.shape[1], -1).shape
        
        embedding, self._indices2 = torch.max(x.view(x.shape[0], x.shape[1], -1), dim=-1)
        x = embedding.view(x.shape[0], -1)
        
        
        return  embedding, x 
    
    def decoder(self, x):
        
        ## This computation because there when doing adaptive avg pooling the pixels in the upsampling have the value of the mean.
        recover_x = torch.zeros(self._final_pooling_shape, device=device, requires_grad=False) #self._global_max_unpool(x, self._indices2, output_size=self._final_pooling_shape)# + torch.zeros((x.shape[0], x.shape[1], *self._encode_recover_shape), device=device)

        # First skip connection
        recover_x[:, :, self._indices2] = x
        x = recover_x.view(self._encode_recover_shape) + self.__x_skip_connection
        x = self._output_deconvolution(x)
        x = torch.relu(x)
        
        # Second skip connection
        x = x + self.__x_init_skip_connections

        for convt in self._list_deconvolutions:
            x = convt(x)
            x = torch.relu(x)

        #print(f"Shape before unpooling: {x.shape}")
        x = self._max_unpooling(x, self._indices, output_size=self._pooling_shape)  # Use output_size to ensure consistency
        #print(f"Shape after unpooling: {x.shape}")
        #x = x + self.__x_init_skip_connections
        x = self._init_deconvolution(x) 
        
        return x
    

    def forward(self, x):

        embedding, x = self.encoder(x=x)
        
        reconstructed_image = self.decoder(x=x)
    
        return nn.functional.sigmoid(reconstructed_image)

    
    

        
        


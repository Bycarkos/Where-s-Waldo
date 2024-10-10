import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import torchvision.models as models

class BaseFeatureExtractor(ABC, nn.Module):
    """
    Abstract base class for feature extractors. 
    Defines the interface for all feature extractors.
    """
    def __init__(self, layer_weights=None):
        super(BaseFeatureExtractor, self).__init__()
        self.layer_weights = layer_weights if layer_weights else []
        
    @abstractmethod
    def _extract_features(self, x):
        """
        Abstract method to extract features from a model.
        Should be implemented by subclasses.
        """
        pass
    
    def forward(self, input, target):
        """
        Compute the perceptual loss between input and target.
        """
        input_features = self._extract_features(input)
        target_features = self._extract_features(target)
        
        perceptual_loss = 0.0
        for i in range(len(input_features)):
            perceptual_loss += self.layer_weights[i] * torch.nn.functional.mse_loss(input_features[i], target_features[i])

        return perceptual_loss


class ResNetFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, layer_weights=None):
        super(ResNetFeatureExtractor, self).__init__(layer_weights)
        
        # Load a pre-trained ResNet model and extract specific layers
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Define the layers we want to extract features from
        self.layers = nn.ModuleList([
            nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool),  # First conv block
            resnet.layer1,  # First residual block
            resnet.layer2,  # Second residual block
            resnet.layer3   # Third residual block
        ]).eval()

        for param in self.layers.parameters():
            param.requires_grad = False  # We don't want to train the ResNet model

        # Set default layer weights if not provided
        if self.layer_weights == []:
            self.layer_weights = [1.0] * len(self.layers)

    def _extract_features(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features



class VGGFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, layer_weights=None):
        super(VGGFeatureExtractor, self).__init__(layer_weights)
        
        # Load a pre-trained VGG model and extract specific layers
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        
        # Extract the first few layers (e.g., up to relu_4_3)
        self.layers = nn.Sequential(*list(vgg.children())[:16]).eval()
        for param in self.layers.parameters():
            param.requires_grad = False  # We don't want to train the VGG model

        # Set default layer weights if not provided
        if self.layer_weights == []:
            self.layer_weights = [1.0] * len(self.layers)

    def _extract_features(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features
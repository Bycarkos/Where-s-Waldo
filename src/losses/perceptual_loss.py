import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self, layer_weights=None):
        super(PerceptualLoss, self).__init__()
        
        # Load a pre-trained VGG model and extract specific layers
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        
        # We will use the first few layers to capture perceptual features
        self.layers = nn.Sequential(*list(vgg.children())[:16]).eval()
        for param in self.layers.parameters():
            param.requires_grad = False  # We don't want to train the VGG model

        # Optionally weight different layers' contributions to the loss
        self.layer_weights = layer_weights if layer_weights else [1.0] * len(self.layers)

    def forward(self, input, target):
        # Ensure input and target are in the right range and have 3 channels
        input_features = self._extract_features(input)
        target_features = self._extract_features(target)
        
        perceptual_loss = 0.0
        for i in range(len(input_features)):
            perceptual_loss += self.layer_weights[i] * torch.nn.functional.mse_loss(input_features[i], target_features[i])

        return perceptual_loss

    def _extract_features(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features



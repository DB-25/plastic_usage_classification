import torch
from torch import nn
import torch.nn.functional as F

# MLP head
class MLP(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_sizes=[128, 64], dropout_probability=[0.5,0.7]):
        super(MLP, self).__init__()
        assert len(hidden_sizes) >= 1 , "specify at least one hidden layer"
        
        self.layers = self.create_layers(in_channels, num_classes, hidden_sizes, dropout_probability)


    def create_layers(self, in_channels, num_classes, hidden_sizes, dropout_probability):
        layers = []
        layer_sizes = [in_channels] + hidden_sizes + [num_classes]
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes)-2:
                layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_probability[i]))
            else:
                layers.append(nn.Softmax(dim=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x.view(x.shape[0], -1)
        out = self.layers(out)
        return out
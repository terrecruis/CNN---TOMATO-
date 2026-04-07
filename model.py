# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================
# MODELLO 1: Fully Connected Neural Network (FCNN)
# Input: immagine 64x64x3 -> appiattita in vettore 12288
# Iperparametri esplorabili: num_hidden_layers, hidden_size, activation
# =============================================================
class TomatoFCNN(nn.Module):
    def __init__(self, num_hidden_layers=2, hidden_size=512, activation='relu', num_classes=10):
        super(TomatoFCNN, self).__init__()
        
        self.activation_name = activation
        input_size = 64 * 64 * 3  # immagini RGB 64x64 appiattite

        # Costruzione dinamica degli strati nascosti
        layers = []
        in_features = input_size
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(0.4))
            in_features = hidden_size

        layers.append(nn.Linear(in_features, num_classes))
        self.network = nn.Sequential(*layers)

    def _get_activation(self, name):
        if name == 'relu':
            return nn.ReLU()
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Activation '{name}' non supportata. Usa: relu, tanh, sigmoid")

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten: (B, 3, 64, 64) -> (B, 12288)
        return self.network(x)


# =============================================================
# MODELLO 2: Convolutional Neural Network (CNN)
# Input: immagine 64x64x3
# Iperparametri esplorabili: n_filters, kernel_size, num_blocks
# =============================================================
class TomatoCNN(nn.Module):
    def __init__(self, n_filters=32, kernel_size=3, num_blocks=3, num_classes=10):
        super(TomatoCNN, self).__init__()
        
        self.num_blocks = num_blocks
        padding = kernel_size // 2  # mantiene dimensione spaziale prima del pooling

        # Costruzione dinamica dei blocchi conv+pool
        conv_blocks = []
        in_channels = 3
        out_channels = n_filters
        for i in range(num_blocks):
            conv_blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            conv_blocks.append(nn.ReLU())
            conv_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
            out_channels = out_channels * 2  # raddoppia i filtri ad ogni blocco

        self.conv_net = nn.Sequential(*conv_blocks)

        # Calcolo dinamico della dimensione spaziale dopo num_blocks pooling:
        # 64 / (2^num_blocks)  ->  es. 3 blocchi: 64/8 = 8
        spatial_size = 64 // (2 ** num_blocks)
        fc_input_dim = in_channels * spatial_size * spatial_size

        self.classifier = nn.Sequential(
            nn.Linear(fc_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.classifier(x)
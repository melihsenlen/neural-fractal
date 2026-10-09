import torch
import torch.nn as nn


class FractalNet(nn.Module):
    def __init__(self, frequencies: int, hidden_dim: int, num_layers: int):
        super().__init__()

        self.frequencies = frequencies
        input_dim = 2 + 4*frequencies

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim # for the hidden layers

        layers.append(nn.Linear(hidden_dim, 3)) # output dimensions: (red, green, blue) => 3
        self.model = nn.Sequential(*layers)

    def fourier_features(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for i in range(1, self.frequencies + 1):
            frequency = 2*torch.pi * 2*i # linear spacing keeps details better in the long run
            features.append(torch.sin(frequency * x))
            features.append(torch.cos(frequency * x))
        return torch.cat(features, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fourier_features(x)
        return self.model(x)
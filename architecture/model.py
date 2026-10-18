import torch
import torch.nn as nn


class NeuralFractal(nn.Module):
    def __init__(self, frequencies: int, hidden_dim: int, num_layers: int):
        super().__init__()

        input_dim = 4*frequencies + 2
        self.frequencies = frequencies

        layers = []
        for layer in range(num_layers):
            layers.append(nn.Linear(input_dim if not layer else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, 3)) # output dimensions: (r, g, b) => 3
        self.model = nn.Sequential(*layers)

    def fourier(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for i in range(1, self.frequencies + 1):
            frequency = 4*torch.pi*i + 2 # linear spacing keeps details better in the long run
            features.append(torch.sin(frequency * x))
            features.append(torch.cos(frequency * x))
        return torch.cat(features, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fourier(x)
        return self.model(x)
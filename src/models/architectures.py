"""PyTorch model architectures."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron for tabular data."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int] = None,
                 dropout: float = 0.1):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TabularTransformer(nn.Module):
    """Transformer encoder for tabular data."""

    def __init__(self, input_dim: int, output_dim: int, d_model: int = 256,
                 nhead: int = 8, num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: (batch, features) -> (batch, 1, d_model) -> pool -> (batch, d_model)
        x = self.embedding(x).unsqueeze(1)
        x = self.encoder(x).squeeze(1)
        return self.head(x)


def build_model(name: str, input_dim: int, output_dim: int, params: dict) -> nn.Module:
    """Factory function to create a model by name."""
    if name == "mlp":
        return MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=params.get("hidden_dims", [128, 64]),
            dropout=params.get("dropout", 0.1),
        )
    elif name == "transformer":
        return TabularTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=params.get("d_model", 256),
            nhead=params.get("nhead", 8),
            num_layers=params.get("num_layers", 4),
            dropout=params.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown model: {name}. Choose from: mlp, transformer")


if __name__ == "__main__":
    print("architectures module ready")

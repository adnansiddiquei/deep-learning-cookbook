import torch
import torch.nn as nn

from dlc.transformers.modules import TransformerEncoderLayer


class Bert(nn.Module):
    def __init__(self, embedding_dim: int, num_transformer_layers: int):
        super().__init__()
        self.embedding_dim = embedding_dim

        assert (
            self.embedding_dim % 64 == 0
        ), '`embedding_dim` must be a multiple of 64. `num_heads` is computed as embedding_dim // 64.'

        self.transformer_layers = nn.ModuleList()

        for _ in range(num_transformer_layers):
            self.transformer_layers.append(
                TransformerEncoderLayer(
                    embedding_dim=embedding_dim,
                    num_heads=embedding_dim // 64,
                    ff_hidden_dim=embedding_dim,
                    dropout=0.1,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.transformer_layers:
            x = layer(x)

        return x

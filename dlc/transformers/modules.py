import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductSelfAttentionBlock(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.embedding_size = embedding_size

        self.values_projection_layer = nn.Linear(embedding_size, embedding_size)
        self.queries_projection_layer = nn.Linear(embedding_size, embedding_size)
        self.keys_projection_layer = nn.Linear(embedding_size, embedding_size)

    def forward(self, x: torch.Tensor):
        """Input tensor x.shape = [N, embedding_size]"""
        assert (
            x.shape[-1] == self.embedding_size
        ), f'Input x has the wrong embedding dimensionality. Recieved: {x.shape[-1]}, expected: {self.embedding_size}'

        """Compute the value projection. This is a simple linear projection of each embedding.
        values.shape = [N, embedding_size]
        """
        values = self.values_projection_layer(x)

        """Compute the queries and keys, these are also simply linear projections of each embedding.
        queries.shape, keys.shape = [N, embedding_size]

        The query and key projection determine how much relevance each embedding has with each other.
        These linear projections directly affect the computation of the attention weights.
        """
        queries = self.queries_projection_layer(x)
        keys = self.keys_projection_layer(x)

        """
        Compute the attention weights. This is the attention paid to each embedding, by every other
        embedding.

        It is scaled by the square root of the embedding size to stabilise training as the dot product
        can have large magnitudes.

        attention_weights.shape = [N, N]
        attention_weights[0] are the attentions paid to embedding 0, by every other embedding.
        attention_weights[0] is also normalised using the softmax, so it sums to 1.
        """
        attention_weights = F.softmax(
            (queries @ keys.T) / self.embedding_size**0.5, dim=-1
        )

        """
        Now we matrix multiply the attention weights with the values to get the output.
        output.shape = [N, embedding_size]

        Each embedding in the output is simply the (attention) weighted sum of each (linearly projected)
        input embedding.

        E.g., If a x[17] "pays" a lot of attention to x[0], then output[0] will contain a large
        portion of x[17].
        """
        output = attention_weights @ values

        return output

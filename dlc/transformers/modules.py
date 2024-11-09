"""
Prince, S.J.D. (2024). Understanding Deep Learning:
 - Ch 12 Transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionBlock(nn.Module):
    """Implementation of scaled dot product self attention."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.values_projection_layer = nn.Linear(embedding_dim, embedding_dim)
        self.queries_projection_layer = nn.Linear(embedding_dim, embedding_dim)
        self.keys_projection_layer = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor):
        """Input tensor x.shape = [N, embedding_dim]"""
        assert (
            x.shape[-1] == self.embedding_dim
        ), f'Input x has the wrong embedding dimensionality. Recieved: {x.shape[-1]}, expected: {self.embedding_dim}'

        """Compute the value projection. This is a simple linear projection of each embedding.
        values.shape = (N, embedding_dim)
        """
        values = self.values_projection_layer(x)

        """Compute the queries and keys, these are also simply linear projections of each embedding.
        queries.shape, keys.shape = [N, embedding_dim]

        The query and key projection determine how much relevance each embedding has with each other.
        These linear projections directly affect the computation of the attention weights.
        """
        queries = self.queries_projection_layer(x)
        keys = self.keys_projection_layer(x)

        """
        Compute the attention weights. This is the attention paid to each embedding, by every other
        embedding.

        It is scaled by the square root of the embedding size to stabilise training as the dot product
        can have large magnitudes. This is where the "scaled" in the name comes from.

        attention_weights.shape = [N, N]
        attention_weights[0] are the attentions paid to embedding 0, by every other embedding.
        attention_weights[0] is also normalised using the softmax, so it sums to 1.

        The dot product here is what makes this block non-linear, and therefore removes the
        requirement of an activation function.
        """
        attention_weights = F.softmax(
            (queries @ keys.T) / self.embedding_dim**0.5, dim=-1
        )

        """
        Now we matrix multiply the attention weights with the values to get the output.
        output.shape = [N, embedding_dim]

        Each embedding in the output is simply the (attention) weighted sum of each (linearly projected)
        input embedding.

        E.g., If a x[17] "pays" a lot of attention to x[0], then output[0] will contain a large
        portion of x[17].
        """
        output = attention_weights @ values

        return output


class MultiHeadSelfAttentionBlock(nn.Module):
    """Implementation of multi head scaled dot product self attention.

    In multi-head self-attention, we compute the attention matrix multiple times.
    This is done by splitting the projection layers a number of times along the embedding dimension,
    and then computing the attentions on split projection layers.
    """

    def __init__(self, embedding_dim: int, num_heads: int):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        assert (
            self.embedding_dim % self.num_heads == 0
        ), '`embedding_dim` must be divisible by `num_heads`.'

        self.head_dim = self.embedding_dim // self.num_heads

        self.values_projection_layer = nn.Linear(embedding_dim, self.embedding_dim)
        self.queries_projection_layer = nn.Linear(embedding_dim, self.embedding_dim)
        self.keys_projection_layer = nn.Linear(embedding_dim, self.embedding_dim)

        self.output_projection_layer = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, x: torch.Tensor):
        """For the sake of explanation, let us assume that
        x.shape = (N, embedding_dim) = (1024, 100)
        self.num_heads = 5.
        """
        assert (
            x.shape[-1] == self.embedding_dim
        ), f'Input x has the wrong embedding dimensionality. Recieved: {x.shape[-1]}, expected: {self.embedding_dim}'

        N, _ = x.shape

        """
        We project the input, but then reshape and transpose the projections to create the multiple
        heads.
        The shape of the projections are    (1024, 100)
        After the .view                     (1024, 5, 20)
        After the .transpose                (5, 1024, 20)
        This creates 5 heads.
        """
        values = (
            self.values_projection_layer(x)
            .view(N, self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        queries = (
            self.queries_projection_layer(x)
            .view(N, self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        keys = (
            self.keys_projection_layer(x)
            .view(N, self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        """
        Quite a lot happening here:
         1. A batchwise matmul is being performed. If A.shape = (b, m, n) and B.shape = (b, n, p)
         then (A @ B).shape = (b, m, p). The matmul is performed over that last 2 dims independently
         for each item in the batch.
         Following the example above
            queries.shape = (5, 1024, 20)
            keys.transpose(-2, -1).shape = (5, 20, 1024)
            attention_weights.shape = (5, 1024, 1024)
         2. Scaled by embedding size.
         3. Then a softmax is computed over the last dim. I.e., 5 * 1024 softmaxes are performed.
        """
        attention_weights = F.softmax(
            (queries @ keys.transpose(-2, -1)) / self.embedding_dim**0.5, dim=-1
        )

        """
        We now compute multiple self attentions.
            attention_weights.shape = (5, 1024, 1024), values.shape = (5, 1024, 20)
            (attention_weights @ values).shape = (5, 1024, 20)
            output.shape = (1024, 100)
        """
        output = (
            (attention_weights @ values).transpose(0, 1).reshape(N, self.embedding_dim)
        )
        output = self.output_projection_layer(output)

        return output


class TransformerEncoderLayer(nn.Module):
    """Implements the Transformer encoder layer as described in:
     - Prince, S.J.D. (2024). Understanding Deep Learning: Ch. 12.4.
     - Vaswani, A., et al. (2017). Attention is all you need. https://arxiv.org/pdf/1706.03762.

    TransformerEncoderLayer is made up on a multi head self attn and feed forward layer.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.multi_head_attn = MultiHeadSelfAttentionBlock(
            embedding_dim=embedding_dim, num_heads=num_heads
        )
        self.layer_norm_1 = nn.LayerNorm((embedding_dim,))

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embedding_dim),
        )
        self.layer_norm_2 = nn.LayerNorm((embedding_dim,))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # self attn block, with residual connection
        x = self.multi_head_attn(x) + self.dropout(x)
        x = self.layer_norm_1(x)

        # feed forward block, with residual connection
        x = self.feed_forward(x) + self.dropout(x)
        x = self.layer_norm_2(x)

        return x

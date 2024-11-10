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

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """For the sake of explanation, let us assume that
        x.shape = (batch_size, sequence_length, embedding_dim) = (256, 512, 100)
        """
        batch_size, sequence_length, embedding_dim = x.shape
        assert (
            embedding_dim == self.embedding_dim
        ), f'Input x has the wrong embedding dimensionality. Recieved: {embedding_dim}, expected: {self.embedding_dim}'

        """
        We project the inputs.
            {values, queries, keys}.shape = (256, 512, 100)

         - values projection is a simple linear projection of each embedding.
         - queries and key projection are also just linear projections.
             - However, semantically these have a different meaning. These projections determine
               how much relevant each embedding has with each other as the attention weights
               are computed from these projections.
        """
        values = self.values_projection_layer(x)
        queries = self.queries_projection_layer(x)
        keys = self.keys_projection_layer(x)

        """
        Now we compute the attention scores. This is the attention paid to each embedding, by every
        other embedding within the same sequence.
            attention_scores.shape = (256, 512, 512)
            attention_scores[0][7].shape = (512,)

         - attention_scores[0][7] are the attentions paid to the 7th embedding in the first
           sequence of the batch, by every other embedding in that sequence.

         - The ` / self.embedding_dim ** 0.5` scaling is to stabilise training. As the dot product
           can have large magnitudes.

         - The dot product here is what makes this block non-linear, so an activation function is
           not needed.
        """
        attention_scores = (queries @ keys.transpose(-2, -1)) / self.embedding_dim**0.5

        """
        Now we apply a mask, if there is one, and then compute the attention_weights using a
        softmax.

        The mask is a tensor of the same shape as the input, and anywhere in the mask where it is
        `0`, we don't want any attention paid to these tokens. By setting them to '-inf', when
        the softmax occurs, these tokens will have 0 attention paid to them.
        """
        if mask is not None:
            assert mask.shape == (batch_size, sequence_length, sequence_length)
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(attention_scores, dim=-1)

        """
        Now we matrix multiply the attention weights with the values to get the output.
            output.shape = (256, 512, 100)

        Each embedding in the output is simply the (attention) weighted sum of each (linearly projected)
        input embedding within the same sequence.

        E.g., If x[0][17] "pays" a lot of attention to x[0][0], then output[0][0] will contain a large
        portion of x[0][17].
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """For the sake of explanation, let us assume that
        x.shape = (batch_size, sequence_length, embedding_dim) = (256, 512, 100)
        self.num_heads = 5.
        """
        batch_size, sequence_length, embedding_dim = x.shape
        assert (
            embedding_dim == self.embedding_dim
        ), f'Input x has the wrong embedding dimensionality. Recieved: {embedding_dim}, expected: {self.embedding_dim}'

        """
        We project the input, but then reshape and transpose the projections to create the multiple
        heads.
        The shape of the projections are    (256, 512, 100)
        After the .view                     (256, 512, 5, 20)
        After the .transpose                (256, 5, 512, 20)
        This creates 5 heads.
        """
        values = (
            self.values_projection_layer(x)
            .view(batch_size, sequence_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        queries = (
            self.queries_projection_layer(x)
            .view(batch_size, sequence_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        keys = (
            self.keys_projection_layer(x)
            .view(batch_size, sequence_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        """
        Quite a lot happening here:
         1. A batchwise matmul is being performed. If A.shape = (b, m, n) and B.shape = (b, n, p)
         then (A @ B).shape = (b, m, p). The matmul is performed over that last 2 dims independently
         for each item in the batch.
         Following the example above
            queries.shape = (256, 5, 512, 20)
            keys.transpose(-2, -1).shape = (256, 5, 20, 512)
            attention_scores.shape = (256, 5, 512, 512)
         2. Scaled by head_dim.
        """
        attention_scores = (queries @ keys.transpose(-2, -1)) / self.head_dim**0.5

        """
        Now we apply a mask, and compute the softmax.

        Note the unsqueeze along the 1st axis. This allows the mask to be broadcasted to every
        head identically.
        """
        if mask is not None:
            assert mask.shape == (batch_size, sequence_length, sequence_length)
            mask = mask.unsqueeze(
                1
            )  # this will yield shape (batch_size, 1, sequence_length, sequence_length)
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(attention_scores, dim=-1)

        """
        We now compute multiple self attentions.
            attention_weights.shape = (256, 5, 512, 512), values.shape = (256, 5, 512, 20)
            (attention_weights @ values).shape = (256, 5, 512, 20)
            output.shape = (256, 512, 100)
        """
        output = (
            (attention_weights @ values)
            .transpose(1, 2)
            .reshape(batch_size, sequence_length, self.embedding_dim)
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
        attn_output = self.multi_head_attn(x)
        x = x + self.dropout(attn_output)
        x = self.layer_norm_1(x)

        # feed forward block, with residual connection
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.layer_norm_2(x)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

from dlc.transformers.modules import TransformerEncoderLayer


class Bert(nn.Module):
    """
    Implementation of the BERT model for the masked language modelling (MLM) pretraining task.

    For simplicity, this model ignores the next sentence prediction (NSP) pretraining task.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_transformer_layers: int,
        vocab_size: int,
        max_sequence_length: int,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_transformer_layer = num_transformer_layers
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length

        assert (
            embedding_dim % 64 == 0
        ), '`embedding_dim` must be a multiple of 64. `num_heads` is computed as embedding_dim // 64.'

        """
        Create a set of token embeddings, which will be used to encode each word in the vocabulary.
        These are initialised to random embeddings, which will move and be learned throughout training
        to acquire some form of semantic meaning.
        """
        self.token_embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )

        """
        Create a set of positional embeddings which will be added onto the inputs during the forward
        pass, so that the model can learn some concept of position. Again, these are randomly
        initialised embeddings and are learned during training.
        """
        self.position_embeddings = nn.Embedding(
            num_embeddings=max_sequence_length, embedding_dim=embedding_dim
        )

        self.transformer_layers = nn.ModuleList()

        for _ in range(num_transformer_layers):
            self.transformer_layers.append(
                TransformerEncoderLayer(
                    embedding_dim=embedding_dim,
                    num_heads=embedding_dim // 64,
                )
            )

        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids.shape = (batch_size, sequence_length)
        assert len(token_ids.shape) == 2

        batch_size, sequence_length = token_ids.shape

        """
        First extract the embeddings for the given token_ids.
            token_embeddings.shape = (batch_size, sequence_length, embedding_size)
        """
        token_embeddings = self.token_embeddings(token_ids)

        """
        Now we need to add on the position embeddings.
        Extracting the embeddings is a slightly convoluted operation but it makes sense if you work
        through it slowly. We just need to create the position_ids for 1 batch, then unsqueeze and
        expand it so it is repeated for every batch.
        """
        position_ids = torch.arange(
            sequence_length, dtype=torch.long, device=token_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand(batch_size, sequence_length)
        position_embeddings = self.position_embeddings(position_ids)

        """
        And here we have the batch of inputs.
            x.shape = (batch_size, sequence_length, embedding_size)
        """
        x = token_embeddings + position_embeddings

        """
        Now we pass the batch through the model.
            Pass through every transformer layer.
            Pass through the Linear output layer to change the embedding_dim -> vocab_size
            Pass through softmax to get token probabilities.
        """
        for layer in self.transformer_layers:
            x = layer(x)

        # logits are the raw, unnormalised probabilities that a given token is in a given class
        logits = self.output_layer(x)
        # logits.shape = (batch_size, sequence_length, vocab_size)

        probabilities = F.softmax(logits, dim=-1)
        # probabilities.shape = (batch_size, sequence_length, vocab_size)

        return probabilities


class BertBase(Bert):
    def __init__(self):
        embedding_dim = 768
        num_transformer_layers = 12
        vocab_size = 3000
        max_sequence_length = 512

        super().__init__(
            embedding_dim=embedding_dim,
            num_transformer_layers=num_transformer_layers,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
        )

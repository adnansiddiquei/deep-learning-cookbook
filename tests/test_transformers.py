import torch
from dlc.transformer.modules import (
    SelfAttentionBlock,
    MultiHeadSelfAttentionBlock,
    TransformerEncoderLayer,
)

batch_size = 128
sequence_length = 256
embedding_dim = 100
num_heads = 5


def test_self_attention_block(device):
    x = torch.randn(batch_size, sequence_length, embedding_dim).to(device)
    self_attention = SelfAttentionBlock(embedding_dim).to(device)
    mask = torch.randint(0, 2, (batch_size, sequence_length, sequence_length)).to(
        device
    )

    with torch.no_grad():
        output = self_attention(x, mask)

    assert (
        output.shape
        == (
            batch_size,
            sequence_length,
            embedding_dim,
        )
    ), f'Expected output shape ({batch_size}, {sequence_length}, {embedding_dim}), but got {output.shape}'


def test_multi_head_self_attention_block(device):
    x = torch.randn(batch_size, sequence_length, embedding_dim).to(device)
    multi_head_attention = MultiHeadSelfAttentionBlock(embedding_dim, num_heads).to(
        device
    )
    mask = torch.randint(0, 2, (batch_size, sequence_length, sequence_length)).to(
        device
    )

    with torch.no_grad():
        output = multi_head_attention(x, mask)

    assert (
        output.shape
        == (
            batch_size,
            sequence_length,
            embedding_dim,
        )
    ), f'Expected output shape ({batch_size}, {sequence_length}, {embedding_dim}), but got {output.shape}'


def test_transformer_encoder_layer(device):
    ff_hidden_dim = 128
    x = torch.randn(batch_size, sequence_length, embedding_dim).to(device)
    transformer_encoder = TransformerEncoderLayer(
        embedding_dim, num_heads, ff_hidden_dim
    ).to(device)

    with torch.no_grad():
        output = transformer_encoder(x)

    assert (
        output.shape
        == (
            batch_size,
            sequence_length,
            embedding_dim,
        )
    ), f'Expected output shape ({batch_size}, {sequence_length}, {embedding_dim}), but got {output.shape}'

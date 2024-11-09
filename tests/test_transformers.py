import torch
from dlc.transformers.modules import (
    SelfAttentionBlock,
    MultiHeadSelfAttentionBlock,
    TransformerEncoderLayer,
)


def test_self_attention_block(device):
    embedding_dim = 100
    x = torch.randn(256, 512, embedding_dim).to(device)
    self_attention = SelfAttentionBlock(embedding_dim).to(device)

    with torch.no_grad():
        output = self_attention(x)

    assert output.shape == (
        256,
        512,
        100,
    ), f'Expected output shape (256, 512, 100), but got {output.shape}'


def test_multi_head_self_attention_block(device):
    embedding_dim = 100
    num_heads = 5
    x = torch.randn(256, 512, embedding_dim).to(device)
    multi_head_attention = MultiHeadSelfAttentionBlock(embedding_dim, num_heads).to(
        device
    )

    with torch.no_grad():
        output = multi_head_attention(x)

    assert output.shape == (
        256,
        512,
        100,
    ), f'Expected output shape (256, 512, 100), but got {output.shape}'


def test_transformer_encoder_layer(device):
    embedding_dim = 100
    num_heads = 5
    ff_hidden_dim = 200
    x = torch.randn(256, 512, embedding_dim).to(device)
    transformer_encoder = TransformerEncoderLayer(
        embedding_dim, num_heads, ff_hidden_dim
    ).to(device)

    with torch.no_grad():
        output = transformer_encoder(x)

    assert output.shape == (
        256,
        512,
        100,
    ), f'Expected output shape (256, 512, 100), but got {output.shape}'

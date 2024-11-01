"""
This file defines
 - `ConvBlock2d` which is a single convolutional building block that can be used to build larger
 convolution networks.
"""

import torch.nn as nn


class ConvBlock2d(nn.Module):
    """A simple 2D convolutional block.

    Includes a convolutional kernel, a LayerNorm and an activation function.

    The purpose of the LayerNorm is to stabilise and improve training dynamics [1]. LayerNorm
    normalises a given sample to mean 0, standard deviation 1.

    [1] Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization [arXiv preprint
        arXiv:1607.06450]. arXiv. https://arxiv.org/abs/1607.06450
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channel_shape: tuple[int, int],  # the shape of the last 2 dimensions
        kernel_size: int = 3,
        padding: int | None = None,
        stride: int = 1,
        dilation: int = 1,
        act: nn.Module = nn.ReLU,
    ):
        super().__init__()
        assert kernel_size % 2 != 0  # ensure kernel_size is odd

        if padding is None:
            # compute the padding if it was not passed in
            padding = kernel_size // 2

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
            ),
            # normalises over the last 3 dims, i.e, over each sample.
            nn.LayerNorm((out_channels, *channel_shape)),
            act(),
        )

    def forward(self, x):
        return self.conv_block(x)

import torch.nn as nn
import pytorch_lightning as L
import numpy as np


def count_parameters(model: nn.Module | L.LightningModule) -> int:
    """Calculate the number of parameters in a model."""
    return np.sum([p.numel() for p in model.parameters()])

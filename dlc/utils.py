import torch.nn as nn
import pytorch_lightning as L
import numpy as np


def count_parameters(model: nn.Module | L.LightningModule) -> int:
    """Calculate the number of parameters in a model."""
    return np.sum([p.numel() for p in model.parameters()])


def count_model_size(model: nn.Module | L.LightningModule) -> int:
    """Returns the model size in MB"""
    model_size_bytes = np.sum(
        [p.element_size() * p.numel() for p in model.parameters()]
    )
    model_size_mb = model_size_bytes / (1024**2)
    return model_size_mb

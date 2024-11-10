import torch.nn as nn
import pytorch_lightning as L
import numpy as np
import httpx
from tqdm import tqdm


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


def download(url: str, output_path: str):
    """Downloads a given file, with a progress bar.

    Args:
        url (str): The URL of the file to download.
        output_path (str): The local path where the downloaded file will be saved.

    Example:
        url = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.en/train-00007-of-00041.parquet"
        output_path = "data/train-00007-of-00041.parquet"
        download(url, output_path)
    """
    with httpx.stream('GET', url, follow_redirects=True) as response:
        response.raise_for_status()

        total = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as file, tqdm(
            total=total, unit_scale=True, unit='B', desc=output_path
        ) as pbar:
            # Iterate over response, in chunks of 8kB and save to the output file
            for chunk in response.iter_bytes(chunk_size=1024 * 8):
                file.write(chunk)
                pbar.update(len(chunk))

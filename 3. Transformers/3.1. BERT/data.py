"""
This script downloads part of the wikimedia/wikipedia dataset from huggingface.

This specifically downloads from the english split (20231101.en), and gets the 7th parquet file
(train-00007-of-00041.parquet). This choice of the 7th file is arbitrary, it is 228MB in size.
"""

from dlc.utils import download
import os

if __name__ == '__main__':
    if not os.path.exists('data'):
        os.mkdir('data')

    url = 'https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.en/train-00007-of-00041.parquet'
    output_path = 'data/train-00007-of-00041.parquet'

    download(url, output_path)

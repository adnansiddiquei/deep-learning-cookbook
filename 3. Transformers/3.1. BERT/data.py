"""
This script downloads the data and pre-processes it ready for training.
"""

import os
import pyarrow.parquet as pq
import unicodedata
from dlc.utils import download
import logging

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s'
)

"""
First we download the data. This is part of the wikimedia/wikipedia dataset from Huggingface.

This specifically downloads from the english split (20231101.en), and gets the 7th parquet file
(train-00007-of-00041.parquet). This choice of the 7th file is arbitrary, it is 228MB in size.
"""


def download_data():
    if not os.path.exists('data'):
        os.mkdir('data')

    if os.path.exists('data/train-00007-of-00041.parquet'):
        logging.info('Data already downloaded.')
        return

    url = 'https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.en/train-00007-of-00041.parquet'
    output_path = 'data/train-00007-of-00041.parquet'

    download(url, output_path)


"""
Now we need to preprocess the data.

In BERT (Devlin, J., et al. (2019)), the only preprocessing they do is remove accents.
We will also train our BERT model as uncased, so we will lowercase everything.
"""


def remove_accents(input_str: str):
    assert isinstance(input_str, str)

    # decompose characters with accents into their base characters plus the combining diacritical marks.
    nfkd_form = unicodedata.normalize('NFKD', input_str)

    # checks if a character is a combining character (diacritical mark), and it is removed if true.
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])


def preprocess():
    input_file = 'data/train-00007-of-00041.parquet'
    output_file = 'data/train-00007-of-00041-preprocessed.parquet'

    with pq.ParquetFile(input_file) as reader:
        schema = reader.schema_arrow
        logging.info(f'\n\nSchema: \n{schema}\n\nMetadata: \n{reader.metadata}\n')

        with pq.ParquetWriter(output_file, schema) as writer:
            for i in range(reader.num_row_groups):
                row_group = reader.read_row_group(i)
                # TODO: preprocess the data by removing the accents and lowercasing.
                writer.write_table(row_group)


if __name__ == '__main__':
    download_data()
    preprocess()

import os
import zipfile
import requests
import tiktoken
import numpy as np
from io import BytesIO
from tqdm import tqdm

from common.types import Dataset, TokenEncoding
from common.constants import DATASET_TOKEN_ENCODING_MAPPING

DATASET_URL = 'http://mattmahoney.net/dc/enwik8.zip'
DATASET_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DATASET_ENCODING = 'utf-8'
TOKEN_ENCODING_DTYPE = np.uint16
TRAIN_VAL_SPLIT_PERCENTAGE = 0.9
TRAIN_SPLIT_FILE_PATH = 'train.npy'
VAL_SPLIT_FILE_PATH = 'val.npy'

if __name__ == '__main__':
    download_dataset_response = requests.get(DATASET_URL, stream=True)
    download_dataset_response.raise_for_status()
    total_bytes_count = int(download_dataset_response.headers.get('Content-Length', 0))
    dataset_buffer = BytesIO()
    with tqdm(total=total_bytes_count, unit='B', unit_scale=True, unit_divisor=1024) as bar:
        for chunk in download_dataset_response.iter_content(chunk_size=DATASET_DOWNLOAD_CHUNK_SIZE):
            if chunk:
                dataset_buffer.write(chunk)
                bar.update(len(chunk))
    dataset_buffer.seek(0)
    with zipfile.ZipFile(dataset_buffer, 'r') as archive:
        dataset_text = archive.read(archive.namelist()[0]).decode(DATASET_ENCODING)

    tokenizer = tiktoken.get_encoding(DATASET_TOKEN_ENCODING_MAPPING[Dataset.ENWIK8].value)
    tokenized_dataset = tokenizer.encode_ordinary(dataset_text)
    
    train_split_size = int(len(tokenized_dataset) * TRAIN_VAL_SPLIT_PERCENTAGE)
    train_split = np.array(tokenized_dataset[:train_split_size], dtype=TOKEN_ENCODING_DTYPE)
    val_split = np.array(tokenized_dataset[train_split_size:], dtype=TOKEN_ENCODING_DTYPE)

    np.save(os.path.join(os.path.dirname(__file__), TRAIN_SPLIT_FILE_PATH), train_split)
    np.save(os.path.join(os.path.dirname(__file__), VAL_SPLIT_FILE_PATH), val_split)

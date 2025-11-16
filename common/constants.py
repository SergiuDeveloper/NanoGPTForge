import os
import torch
import torch.nn as nn

from common.types import Model, Optimizer, Scheduler, Dataset, TokenEncoding
from models.nanogpt import NanoGPT

DATA_FOLDER_PATH = 'data'
TRAIN_SPLIT_NAME = 'train'
VAL_SPLIT_NAME = 'val'
CHECKPOINTS_FOLDER_PATH = 'checkpoints'
SEED = 1337

MODEL_MAPPING: dict[Model, type[NanoGPT]] = {
    Model.NANOGPT: NanoGPT
}
MODEL_CONFIG_MAPPING: dict[Model, type] = {
    Model.NANOGPT: NanoGPT.Config
}

OptimizerType = type[torch.optim.AdamW]
DEFAULT_OPTIMIZER = Optimizer.ADAMW
OPTIMIZER_MAPPING: dict[Optimizer, OptimizerType] = {
    DEFAULT_OPTIMIZER: torch.optim.AdamW
}

SchedulerType = type[torch.optim.lr_scheduler.CosineAnnealingLR]
DEFAULT_SCHEDULER = Scheduler.COSINE_ANNEALING_LR
SCHEDULER_MAPPING: dict[Scheduler, SchedulerType] = {
    DEFAULT_SCHEDULER: torch.optim.lr_scheduler.CosineAnnealingLR
}

DATASET_TOKEN_ENCODING_MAPPING: dict[Dataset, TokenEncoding] = {
    Dataset.ENWIK8: TokenEncoding.GPT2
}

TOKEN_ENCODING_SPECIAL_TOKENS: dict[TokenEncoding, set[str]] = {
    TokenEncoding.GPT2: set(['<|endoftext|>'])
}

missing_models = [model_name.value for model_name in Model if model_name not in MODEL_MAPPING]
missing_model_configs = [model_name.value for model_name in Model if model_name not in MODEL_CONFIG_MAPPING]
missing_optimizers = [optimizer_name.value for optimizer_name in Optimizer if optimizer_name not in OPTIMIZER_MAPPING]
missing_schedulers = [scheduler_name.value for scheduler_name in Scheduler if scheduler_name not in SCHEDULER_MAPPING]
missing_folder_datasets = [dataset_name.value for dataset_name in Dataset if not os.path.exists(os.path.join(DATA_FOLDER_PATH, dataset_name.value))]
missing_token_encoding_datasets = [dataset_name.value for dataset_name in Dataset if dataset_name not in DATASET_TOKEN_ENCODING_MAPPING]
unused_token_encodings = [token_encoding.value for token_encoding in TokenEncoding if token_encoding not in DATASET_TOKEN_ENCODING_MAPPING.values()]
missing_special_tokens_token_encodings = [token_encoding.value for token_encoding in TokenEncoding if token_encoding not in TOKEN_ENCODING_SPECIAL_TOKENS]
assert len(missing_models) == 0, f'Models not defined in mapping: {missing_models}'
assert len(missing_model_configs) == 0, f'No config type defined for models: {missing_model_configs}'
assert len(missing_optimizers) == 0, f'Optimizers not defined in mapping: {missing_optimizers}'
assert len(missing_schedulers) == 0, f'Schedulers not defined in mapping: {missing_schedulers}'
assert len(missing_folder_datasets) == 0, f'No folder found for datasets: {missing_folder_datasets}'
assert len(missing_token_encoding_datasets) == 0, f'No token encoding defined for datasets: {missing_token_encoding_datasets}'
assert len(unused_token_encodings) == 0, f'Unused token encodings: {unused_token_encodings}'
assert len(missing_special_tokens_token_encodings) == 0, f'No special tokens defined for token encodings: {missing_special_tokens_token_encodings}'
